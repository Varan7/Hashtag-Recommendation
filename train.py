import pickle as pkl 
import numpy as np
import tensorflow as tf 
from tensorflow.contrib import rnn
from copynet import CopyNetWrapper
import os


with open("/data/shrey/copynet/pickle_data.txt",'r') as file:
	dic = pkl.load(file)
with open("/data/shrey/copynet/pickle_embed.txt",'r') as file:
	dic2 = pkl.load(file)

embeddings=dic2['embeddings']
word2int=dic2['word2int']
delta = dic2['delta']
words = dic2['words']
print len(words)
print len((delta).union(words))
total=dic['total']
vocab_size=len(word2int)
p=dic['max_len2']




def next_element(k):
	n = vocab_size
	dici={}
	x=np.load('/data/shrey/copynet/x/batch_'+str(k)+'.npy')
	t=np.load('/data/shrey/copynet/t/batch_'+str(k)+'.npy')
	y=np.load('/data/shrey/copynet/y/batch_'+str(k)+'.npy')
	z=np.load('/data/shrey/copynet/z/batch_'+str(k)+'.npy')
	len_keys=np.load('/data/shrey/copynet/ln_keys/batch_'+str(k)+'.npy')
	len_docs=np.load('/data/shrey/copynet/ln_docs/batch_'+str(k)+'.npy')
	return x,y,len_docs,len_keys,z,t




## HyperParameters
rnn_size = 64
batch_size = 4
nm_epochs =4


b=tf.placeholder('int32',[batch_size])
d=tf.placeholder('int32',[batch_size])
a = tf.placeholder('int32',[batch_size,dic['max_len']])
g = tf.placeholder('int32',[batch_size,dic['max_len']])
c = tf.placeholder('int32',[batch_size,None])
f = tf.placeholder('int32',[batch_size,dic['max_len2']])


encoder_emb_inp = tf.nn.embedding_lookup(embeddings, a)
decoder_emb_inp = tf.nn.embedding_lookup(embeddings, f)
encoder_cell = rnn.GRUCell(rnn_size)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_emb_inp,sequence_length=b,dtype=tf.float32)
decoder_cell = rnn.GRUCell(rnn_size)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_size,encoder_outputs,memory_sequence_length=b)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=rnn_size)
initial_state = decoder_cell.zero_state(tf.shape(a)[0], dtype=tf.float32).clone(cell_state=encoder_state)
decoder_cell = CopyNetWrapper(decoder_cell, encoder_outputs, g,25000,vocab_size,sequence_length=b)
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,d)
initial_state = decoder_cell.zero_state(tf.shape(a)[0], dtype=tf.float32).clone(cell_state=initial_state)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper,initial_state,output_layer=None)
outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output
def train():
	all_saver = tf.train.Saver()
	labels =tf.one_hot(c,25000,on_value=1.0, off_value=0.0,axis=-1)
	crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
	crossent=tf.reduce_sum(crossent,-1)
	print np.shape(crossent)
	weights=tf.sequence_mask(d,dtype=tf.float32)
	loss = (tf.reduce_sum(crossent *weights)/batch_size)
	params = tf.trainable_variables()
	print params
	gradients = tf.gradients(loss, params)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
	optimizer = tf.train.AdamOptimizer(0.0005)
	update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
	
	with tf.Session() as sess:
		print 'yay'
		sess.run(tf.global_variables_initializer())
		all_saver.restore(sess, '/data/shrey/copynet/model' + '/data-all')
		for epoch in xrange(2,nm_epochs+1):
			print 'alpha'
			epoch_loss = 0
			for no in xrange(No_of_Batches):
				a1,a2,a3,a4,a6,a5=next_element(no)
				a2 = np.array(a2)
				a2 = a2[:,:max(a4)]
				a2= np.array(a2)
				_,co = sess.run([update_step,loss],feed_dict={a:a1,b:a3,c:a2,d:a4,f:a6,g:a5})
				del(a1)
				del(a2)
				del(a3)
				del(a4)
				del(a5)
				del(a6)
				epoch_loss += co
				print epoch, no ,epoch_loss
			print("Epoch",epoch,'completed out of',nm_epochs,'loss:',epoch_loss)
			all_saver.save(sess, '/data/shrey/copynet/model' + '/data-all')

train()