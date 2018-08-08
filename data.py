import re
import pickle as pkl 
import numpy as np
import sys
dic_keys ={}
len_docs = []


##DATA_PREPROCESSING

with open("/users/misc/rharish/hash/copynet/pickle_embed.txt",'r') as file:
	dic2 = pkl.load(file)
word2int=dic2['word2int']
int2word=dic2['int2word']
vocab_size = len(word2int.keys())
delta = dic2['delta']
words = dic2['words']

keys=dic2['keys']
data2=dic2['data2']

for i in xrange(len(data2)):
	file = str(i)
	string=keys[i].split(';')
	for j,a in enumerate(string):
		a = re.sub('[^a-z\ ]+',' ',a.lower())
		a = re.sub('\ +',' ',a)
		if len(a.split())!=0 and len(a.split())<5:
			dic_keys[file+'_'+str(j)]=a
dic['dic_keys']=dic_keys
max_len=0
temp ={}



for i,cont in enumerate(data2):
	max_len=max(max_len,len(cont.split()))
	temp[i]=len(cont.split())

dic['max_len']=max_len

def dic_docs(i,count,dici):
	gamma = data2[i].split()
	lis = np.ones(max_len)*word2int['EOS']
	for j,word in enumerate(gamma):
		if word in words:
			lis[j]=(word2int[word])
			if lis[j]>vocab_size:
				lis[j]=word2int['UNK']
		else:
			lis[j]=(word2int['UNK'])
	return lis,count,dici

def tdic_docs(i,count,dici):
	gamma = data2[i].split()
	tlis = np.ones(max_len)*word2int['EOS']
	for j,word in enumerate(gamma):
		if word in words:
			tlis[j]=(word2int[word])
			if tlis[j]>vocab_size-1:
				if word in dici.keys():
					tlis[j]=dici[word]
				else:
					dici[word]=count
					count+=1
					tlis[j]=count-1
		else:
			if word in dici.keys():
				tlis[j]=dici[word]
			else:
				dici[word]=count
				count+=1
				tlis[j]=count-1
	return tlis,count,dici

dic['temp']=temp


t = []
x = []
y = []
z = []
dici={}
len_keys=[]
max_len2=0
for doc in dic_keys.keys():
	key = dic_keys[doc]
	i=len(key.split())
	i+=1
	max_len2=max(max_len2,i)
print max_len2
total = 0
count = vocab_size-1
print len(dic_keys.keys())
for j,doc in enumerate(dic_keys.keys()):
	
	key = dic_keys[doc]
	document = int(re.sub('_[0-9]+$','',doc))
	len_docs.append(temp[document])
	lis = np.ones(max_len2)*word2int['EOS']
	liss=np.ones(max_len2)*word2int['EOS']
	lis[0]=(word2int['SOS'])
	for i,word in enumerate(key.split()):
		if word in words:
			liss[i]=(word2int[word])
			if liss[i]>vocab_size:
				if word in dici.keys():
					liss[i]=dici[word]
				else:
					dici[word]=count
					count+=1
					liss[i]=dici[word]
		else:
			if word in dici.keys():
				liss[i]=dici[word]
			else:
				dici[word]=count
				count+=1
				liss[i]=dici[word]
	for i,word in enumerate(key.split()):
		if word in words:
			lis[i+1]=(word2int[word])
			if lis[i+1]>vocab_size:
				lis[i+1]=word2int['UNK']
		else:
			lis[i+1]=(word2int['UNK'])
	i+=2
	len_keys.append(i)
	x.append(dic_docs(document,count,dici)[0])
	count = dic_docs(document,count,dici)[1]
	dici = dic_docs(document,count,dici)[2]
	t.append(tdic_docs(document,count,dici)[0])
	count = dic_docs(document,count,dici)[1]
	dici = dic_docs(document,count,dici)[2]
	y.append(liss)
	z.append(lis)


# 4 is the batch_size, ypu can take any batch_size which suits your memory limits.

	if (j+1)%4==0:
		x=np.array(x)
		t=np.array(t)
		y=np.array(y)
		z=np.array(z)
		len_keys=np.array(len_keys)
		len_docs=np.array(len_docs)
		total = int(j/4)
		print int(j/4)
		np.save('/data/shrey/copynet/x/batch_'+str(int(j/4)),x)
		np.save('/data/shrey/copynet/t/batch_'+str(int(j/4)),t)
		np.save('/data/shrey/copynet/y/batch_'+str(int(j/4)),y)
		np.save('/data/shrey/copynet/z/batch_'+str(int(j/4)),z)
		np.save('/data/shrey/copynet/ln_keys/batch_'+str(int(j/4)),len_keys)
		np.save('/data/shrey/copynet/ln_docs/batch_'+str(int(j/4)),len_docs)
		t = []
		x = []
		y = []
		z = []
		len_keys=[]
		len_docs=[]
		dici={}
		count = vocab_size-1


dic['total']=total
dic['max_len2']=max_len2
with open("/data/shrey/copynet/pickle_data.txt",'w') as file:
	pkl.dump(dic,file)