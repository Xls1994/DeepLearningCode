# -*- coding: utf-8 -*-
'''
author:yangyl
skip-gram model implement using Keras model
'''
from keras.layers import Dense,Reshape
from keras.layers.merge import Dot,dot
from keras.layers import Embedding
from keras.models import Sequential,Model
from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams
vocab_size =5000
embed_size =100

word_model =Sequential()
word_model.add(Embedding(vocab_size,embed_size,
                         embeddings_initializer="glorot_uniform",
                         input_length=1))
word_model.add(Reshape((embed_size,)))

context_model =Sequential()
context_model.add(Embedding(vocab_size,embed_size,
                            embeddings_initializer="glorot_uniform",
                            input_length=1)
                  )
context_model.add(Reshape((embed_size,)))


match =dot([word_model.output,context_model.output],1)
output =Dense(1,kernel_initializer='glorot_uniform'
                ,activation='sigmoid')(match)

model =Model(inputs=[word_model.input,context_model.input],outputs=output)

model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
def loadData():
    '''
    I love green eggs and ham.
    (context,word):
    ([I,green],love)
    ([love,eggs],green)
    ([green,and],eggs)
    -------------->
    (love,I) 1
    (love,green) 1
    :return:
    '''
    text ='I love green eggs and ham .'
    tokenizer =Tokenizer()
    tokenizer.fit_on_texts([text])
    word2id =tokenizer.word_index
    id2word ={v:k for k,v in word2id.items()}
    wids =[word2id[w]for w in text_to_word_sequence(text,split=' ')]
    pairs,labels =skipgrams(wids,len(word2id))
    print (len(pairs),len(labels))
    for i in range(10):
        print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
            id2word[pairs[i][0]], pairs[i][0],
            id2word[pairs[i][1]], pairs[i][1],
            labels[i]))

if __name__ == '__main__':
    loadData()