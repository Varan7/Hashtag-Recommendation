import pickle as pkl 
import numpy as np
import tensorflow as tf 
from beam import BeamSearchDecoder
import re
from tensorflow.contrib import rnn
import os
from copynet import CopyNetWrapper
import string

with open("/data/shrey/copynet/pickle_data.txt",'r') as file:
	dic = pkl.load(file)
with open("/data/shrey/copynet/pickle_embed.txt",'r') as file:
	dic2 = pkl.load(file)




embeddings=dic2['embeddings']
word2int=dic2['word2int']
int2word=dic2['int2word']


word2int1 = dict(word2int)
words = word2int.keys()
int2word1 = dict(int2word)
delta = []
x = []
t = []
length =0

## Loading the text file for which we will be suggesting the hashtags
with open('/data/shrey/copynet/text.txt','r') as file:
	alpha = file.read().splitlines()
	for beta in alpha:
		gamma = re.sub('[^a-z\ ]+',' ',beta.lower())
		gamma=re.sub('\ +',' ',gamma)
		for word in gamma.split():
			if word in word2int:
				x.append(word2int[word])
			elif word not in word2int:
				delta.append(word)
				x.append(word2int['UNK'])
for word in delta:
	if word not in word2int.keys():
		word2int1[word]=n1
		int2word1[n1]=word
		n1+=1
	elif word2int[word]>vocab_size-1:
		word2int1[word]=n1
		int2word1[n1]=word
		n1+=1
with open('/data/shrey/copynet/text.txt','r') as file:
	alpha = file.read().splitlines()
	for beta in alpha:
		gamma = re.sub('[^a-z\ ]+',' ',beta.lower())
		gamma=re.sub('\ +',' ',gamma)
		for word in gamma.split():
			length+=1
			t.append(word2int1[word])
			

assert len(x)==len(t)
len_docs=[length]

x = [x]
x = np.array(x)
t = [t]
t = np.array(t)


len_docs=np.array(len_docs)
beam_width =5
rnn_size = 64
batch_size = np.shape(len_docs)[0]
L1=tf.placeholder('int32',[batch_size])
X = tf.placeholder('int32',[batch_size,length])
T = tf.placeholder('int32',[batch_size,length])




def nn(x,len_docs,t):
	encoder_emb_inp = tf.nn.embedding_lookup(embeddings, x)
	encoder_cell = rnn.GRUCell(rnn_size)
	encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_emb_inp,sequence_length=len_docs,dtype=tf.float32)
	tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
	tiled_sequence_length = tf.contrib.seq2seq.tile_batch(len_docs, multiplier=beam_width)
	tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
	tiled_t = tf.contrib.seq2seq.tile_batch(t,multiplier=beam_width)
	start_tokens = tf.constant(word2int['SOS'], shape=[batch_size])
	decoder_cell = rnn.GRUCell(rnn_size)
	attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_size,tiled_encoder_outputs,memory_sequence_length=tiled_sequence_length)
	decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=rnn_size)
	initial_state = decoder_cell.zero_state(batch_size*beam_width, dtype=tf.float32).clone(cell_state=tiled_encoder_final_state)
	decoder_cell = CopyNetWrapper(decoder_cell, tiled_encoder_outputs, tiled_t,len(set(delta).union(words)),vocab_size,sequence_length=tiled_sequence_length)
	initial_state = decoder_cell.zero_state(batch_size*beam_width, dtype=tf.float32).clone(cell_state=initial_state)
	decoder = BeamSearchDecoder(cell=decoder_cell,embedding=embeddings,start_tokens=start_tokens,end_token=word2int['EOS'],initial_state=initial_state,beam_width=beam_width,output_layer=None,length_penalty_weight=0.0)
	outputs,j,k = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=2)
	logits = outputs.predicted_ids
	return logits


def answer():
	logits = nn(X,L1,T)
	print tf.trainable_variables()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, '/data/shrey/copynet/model' + '/data-all')
		array = (sess.run(logits,feed_dict={X:x,L1:len_docs,T:t}))
		a=np.shape(array)[1]
		b=np.shape(array)[2]
		for j in range(b): 
			for i in range(a):
				c=int(array[0,i,j])
				print int2word1[c]
			print'\n'

answer()