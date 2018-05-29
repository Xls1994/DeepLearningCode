import numpy as np 
np.random.seed(13)

from keras.models import Sequential,Model
from keras.layers import Embedding,Reshape,Activation,Input
from keras.layers.merge import Dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
# load data
corpus =open("alice",'r').readlines()
corpus =[s.strip() for s in corpus if s.count(" ")>=2]
print corpus
tokenizer =Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_num =len(tokenizer.word_index)+1
embed_dim =128
def train_skip():
    # build model
    w_input =Input(shape=(1,),dtype='int32')
    embed =Embedding(vocab_num,embed_dim)

    w =embed(w_input)
    c_input =Input(shape=(1,))
    c =embed(c_input)

    o =Dot(axes=2)([w,c])
    o =Reshape((1,),input_shape =(1,1))(o)
    o =Activation('sigmoid')(o)
    Skipgram =Model(inputs=[w_input,c_input],outputs =o)
    Skipgram.summary()
    Skipgram.compile(loss ='binary_crossentropy',
    optimizer='Adam')

    for _ in range(5):
        loss =0
        for i,doc in enumerate(tokenizer.texts_to_sequences(corpus)):
            data,lables =skipgrams(sequence=doc,vocabulary_size=vocab_num,
            window_size=5,negative_samples=5)
            x=[np.array(x)for x in zip(*data)]
            y =np.array(lables,dtype=np.int32)
            if x:
                loss+=Skipgram.train_on_batch(x,y)
    print loss

    f =open('vectors.txt','w')
    f.write('{} {}\n'.format(vocab_num-1,embed_dim))
    vectors =Skipgram.get_weights()[0]
    for word,i in tokenizer.word_index.iteritems():
        f.write("{} {}\n".format(word,' '.join(map(str,list(vectors[i,:])))))
    f.close()
# train_skip()

from gensim.models.word2vec import Word2Vec
model = Word2Vec.load_word2vec_format('./vectors.txt', binary=False)
print model.wv