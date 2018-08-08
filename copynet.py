import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util

class CopyNetWrapperState(
    collections.namedtuple("CopyNetWrapperState", ("prob_c", "last_ids","cell_state"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(CopyNetWrapperState, self)._replace(**kwargs))

class CopyNetWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, encoder_states, encoder_input_ids, vocab_size,
            gen_vocab_size=None, encoder_state_size=None, initial_cell_state=None,sequence_length=None,name=None):
        """
        Args:
            cell:
            encoder_states:
            encoder_input_ids:
            tgt_vocab_size:
            gen_vocab_size:
            encoder_state_size:
            initial_cell_state:
        """
        super(CopyNetWrapper, self).__init__(name=name)
        self._cell = cell
        self._vocab_size = vocab_size
        self._gen_vocab_size = gen_vocab_size or vocab_size
        self._sequence_length=sequence_length
        self._encoder_input_ids = encoder_input_ids
        self._encoder_states = encoder_states
        if encoder_state_size is None:
            encoder_state_size = self._encoder_states.shape[-1].value
            if encoder_state_size is None:
                raise ValueError("encoder_state_size must be set if we can't infer encoder_states last dimension size.")
        self._encoder_state_size = encoder_state_size

        self._initial_cell_state = initial_cell_state
        self._copy_weight = tf.get_variable('CopyWeight', [self._encoder_state_size , self._cell.output_size])
        self._projection = tf.layers.Dense(self._gen_vocab_size, use_bias=False, name="layer")

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, CopyNetWrapperState):
            raise TypeError("Expected state to be instance of CopyNetWrapperState. "
                      "Received type %s instead."  % type(state))
        last_ids = state.last_ids
        prob_c = state.prob_c
        # print('mc',np.shape(prob_c))
        cell_state = state.cell_state

        mask = tf.cast(tf.equal(tf.expand_dims(last_ids, 1),  self._encoder_input_ids), tf.float32)
        mask = mask*tf.sequence_mask(self._sequence_length,tf.shape(mask)[-1],dtype=tf.float32)
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask = tf.where(tf.less(mask_sum, 1e-7), mask, mask / tf.expand_dims(mask_sum, 1))
        rou = mask * prob_c
        selective_read = tf.einsum("ijk,ij->ik", self._encoder_states, rou)
        inputs = tf.concat([inputs, selective_read], 1)

        outputs, cell_state = self._cell(inputs, cell_state, scope)
        generate_score = self._projection(outputs)

        copy_score = tf.einsum("ijk,km->ijm", self._encoder_states, self._copy_weight)
        copy_score = tf.nn.tanh(copy_score)

        copy_score = tf.einsum("ijm,im->ij", copy_score, outputs)
        encoder_input_mask = tf.one_hot(self._encoder_input_ids, self._vocab_size)
        expanded_copy_score = tf.einsum("ijn,ij->ij", encoder_input_mask, copy_score)
        expanded_copy_score=expanded_copy_score*tf.sequence_mask(self._sequence_length,tf.shape(expanded_copy_score)[-1],dtype=tf.float32)

        prob_g = generate_score
        prob_c = expanded_copy_score
        # print('mc',np.shape(prob_c))
#        mixed_score = tf.concat([generate_score, expanded_copy_score], 1)
#        probs = tf.nn.softmax(mixed_score)
#        prob_g = probs[:, :self._gen_vocab_size]
#        prob_c = probs[:, self._gen_vocab_size:]

        prob_c_one_hot = tf.einsum("ijn,ij->in", encoder_input_mask, prob_c)
        prob_g_total = tf.pad(prob_g, [[0, 0], [0, self._vocab_size - self._gen_vocab_size]])
        outputs = prob_c_one_hot + prob_g_total
        last_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        #prob_c.set_shape([None, self._encoder_state_size])
        state = CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)
        # print np.shape(outputs),np.shape(state)
        return outputs, state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return CopyNetWrapperState(cell_state=self._cell.state_size, last_ids=tf.TensorShape([]),
            prob_c = tf.shape(self._encoder_states)[1])

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            last_ids = tf.zeros([batch_size], tf.int32) - 1
            # print ('miljaa',tf.shape(self._encoder_states)[2])
            prob_c = tf.zeros([batch_size, tf.shape(self._encoder_states)[1]], tf.float32)
            # print ('bhak',cell_state)
            return CopyNetWrapperState(cell_state=cell_state, last_ids=last_ids, prob_c=prob_c)