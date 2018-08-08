import os
import re
import pickle as pkl 
import numpy as np
import tensorflow as tf
import math
import collections
import random
import json

os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
dictionary ={} # Will save all the arrays and dictionaries in this dictionary for further use.

##HyperParameters for Embedding training
dim_word_emb = 250  # Size of word embedding
num_sampled = 80
batch_size = 1024
nm_epochs = 7




### Data-Loading###
data2 = []
keys = []
data1=[]
with open('/data/shrey/copynet/kp20k_training.json',mode = "r")as f :
	text = f.read()
	for content1 in text.split('\n')[:-1]:
		data1.append(json.loads(content1.lower()))
abstracts = []
title= []
keywords = []
for i,z in enumerate(data1):
	abstracts.append(z['abstract'])
	title.append(z['title'])
	keywords.append(z['keyword'])
for y in range(len(abstracts)):
	alpha = abstracts[y]+' '+title[y]
	alpha = re.sub('[^a-z\ ]+',' ',alpha.lower())
	alpha = re.sub('\ +',' ',alpha)
	data2.append(alpha)
for y in range(len(keywords)):
	keys.append(''.join(keywords[y]))


keys=dictionary['keys']
data2=dictionary['data2']

words = []
for i in range(len(data2)):
	string = data2[i]
	for word in string.split():
		words.append(word)
	key_list=keys[i].split(';')
	for a in key_list:
		for word in a.lower().split():
			words.append(word)

## Taking a vocubalary of 30000 words.

qw = {}
for word in words:
	if word in qw:
		qw[word]+=1
	else:
		qw[word]=1
popular_words = sorted(qw, key = qw.get, reverse = True)
phi = popular_words[min(30000,int(len(set(words))*0.6)):min(50000,int(len(set(words))))]
popular_words = popular_words[:min(30000,int(len(set(words))*0.6))]
phi = set(phi)
if 'UNK' not in popular_words:
	popular_words.append('UNK')
if 'SOS' not in popular_words:
	popular_words.append('SOS')
if 'EOS' not in popular_words:
	popular_words.append('EOS')
for word in phi:
	delta.append(word)

words=popular_words
print len(popular_words)
delta = set(delta)
dictionary['words']=words



word2int = {}
int2word = {}
vocab_size = len(words)
for i in range(vocab_size):
	word = words[i]
	word2int[word] =i
	int2word[i] = word

x = []
y = []
print len(data2)
dictionary['word2int']=word2int
dictionary['int2word']=int2word




data = []
for i in range(len(data2)):
	print i
	string = data2[i].split()
	data.append(word2int['SOS'])
	for word in string:
		if word in words:
			data.append(word2int[word])
		else:
			data.append(word2int['UNK'])
	data.append(word2int['EOS'])
print len(data)


data_index = 0
def generate_batch(batch_size,num_skips):
	assert batch_size % num_skips == 0
	assert num_skips <= 2
	global data_index
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 3  # [ 1 target 1 ]
	buffer = collections.deque(maxlen=span)  
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = 1  # target label at the center of the buffer
		targets_to_avoid = [1]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[1]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels


a = tf.placeholder(tf.int32, shape=[None])
b = tf.placeholder(tf.int32, shape=[None, 1])
embeddings = tf.Variable(tf.random_uniform([vocab_size,dim_word_emb], -1.0, 1.0))
nce_weights = tf.Variable(tf.truncated_normal([vocab_size, dim_word_emb],stddev=1.0/math.sqrt(dim_word_emb)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))
embed = tf.nn.embedding_lookup(embeddings,a)
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=b,inputs=embed,num_sampled=num_sampled,num_classes=vocab_size))
optimizer = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(1,nm_epochs+1):
		epoch_loss = 0
		for j in range(int((len(data))/batch_size)):
			print epoch_loss*int((len(data))/batch_size)
			a1,a2=generate_batch(batch_size,2)
			_,c = sess.run([optimizer,loss],feed_dictionaryt={a:a1,b:a2})
			epoch_loss += c
		print("Epoch",epoch,'completed out of',nm_epochs,'loss:',epoch_loss)
	dictionary['embeddings']=sess.run(embeddings)


##Saving the Dictionary
with open("/data/shrey/copynet/pickle_embed.pkl",'w') as file:
	pkl.dump(dictionary,file)