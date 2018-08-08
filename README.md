This is the encoder decoder model which involves attention mechanism and the copyong mechanism for the trending hashtags. It is implmented using the following paper 

Deep Keyphrase Generation by
Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky, Yu Chi
You can find it here: https://arxiv.org/pdf/1704.06879.pdf 


The repository has six python files.
1. Embedding.py; is used to train the word embeddigns. Much of the code is taken from the tutorials which tensorflow provides.
2. Data.py: is used to pre-process the data and arrange them in batches which have following six lists
3. Train.py: is used to train the  tensorflow model a
4. copynet.py: contains the code for the copyuing mechanism used. Much of this has been taken from https://github.com/lspvic/CopyNet
5. Beam.py: This is the exact code which tensorlow provides excpet in few places where we had to remove the softmax function used in the logits, because copynet.py was already doing that for us.
6. Infer.py: This is used to load the tensorflow model and infer the results.