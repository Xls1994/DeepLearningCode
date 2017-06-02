# -*- coding:utf-8 -*-
import numpy as np
# 自定义损失和多输出的LSTM
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Input,Lambda
from keras.layers.merge import concatenate
from keras.datasets import imdb
import tensorflow as tf
from keras.losses import mean_squared_error

max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32
def imdbload(path='imdb.npz', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):

    f = np.load(path)
    x_train = f['x_train']
    labels_train = f['y_train']
    x_test = f['x_test']
    labels_test = f['y_test']
    f.close()

    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(labels_train)

    np.random.seed(seed * 2)
    np.random.shuffle(x_test)
    np.random.seed(seed * 2)
    np.random.shuffle(labels_test)

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        new_xs = []
        new_labels = []
        for x, y in zip(xs, labels):
            if len(x) < maxlen:
                new_xs.append(x)
                new_labels.append(y)
        xs = new_xs
        labels = new_labels
    if not xs:
        raise ValueError('After filtering for sequences shorter than maxlen=' +
                         str(maxlen) + ', no sequence was kept. '
                         'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x] for x in xs]
    else:
        new_xs = []
        for x in xs:
            nx = []
            for w in x:
                if w >= num_words or w < skip_top:
                    nx.append(w)
            new_xs.append(nx)
        xs = new_xs

    x_train = np.array(xs[:len(x_train)])
    y_train = np.array(labels[:len(x_train)])

    x_test = np.array(xs[len(x_train):])
    y_test = np.array(labels[len(x_train):])

    return (x_train, y_train), (x_test, y_test)
def loadData():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdbload(maxlen=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print("Pad sequences (samples x time)")
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train,y_train,x_test,y_test

def mean_squared_errors(y_true,y_pred):
    import keras.backend as K
    return K.mean(K.square(y_pred-y_true),axis=-1)
def lossfunction(args):
    import keras.backend as K
    enout,cnout =args
    out =K.mean(K.square(enout-cnout),axis=-1)
    return out
def E_C_lstm():

    English_input = Input(shape=(maxlen,))
    Engem =Embedding(max_features, 128, input_length=maxlen)(English_input)

    Chinese_input =Input(shape=(maxlen,))
    CHem =Embedding(max_features,64,input_length=maxlen)(Chinese_input)

    ENlstm =LSTM(64,dropout=0.5)(Engem)
    CHlstm =LSTM(64,dropout=0.5)(CHem)

    ENout =Dense(2,activation='sigmoid',name='ENout')(ENlstm)
    CHout =Dense(2,activation='sigmoid',name='CHout')(CHlstm)
    loss_out =Lambda(lossfunction,output_shape=(2,),name='losss')([ENout,CHout])

    out =concatenate(inputs=[ENout,CHout])
    out =Dense(2,activation='softmax',name='Finout')(out)

    model =Model(inputs=[English_input,Chinese_input], outputs=[out,ENout,CHout,loss_out])
    model.summary()
    return  model


model =E_C_lstm()
model.compile(optimizer='sgd',loss={'Finout':'categorical_crossentropy','losss':lambda x ,y:y,'ENout':'categorical_crossentropy',
                                 'CHout':'categorical_crossentropy'})
from keras.utils import plot_model
x_train,y_train,x_test,y_test =loadData()
from keras.utils.np_utils import to_categorical
y_train =to_categorical(y_train,2)
model.fit([x_train,x_train],[y_train,y_train,y_train,y_train],batch_size=32,epochs=10)
plot_model(Mo,'mo.png')
# import json
# from keras.models import model_from_json
# with open('model.json', 'r') as f:
#     jsonfile =json.load(f)
#     print 'load json file...'
# model =model_from_json(json_string=jsonfile)

# # try using different optimizers and different optimizer configs
# model.compile(optimizer='adam',loss= 'binary_crossentropy', loss_weights=[1,0.5,0.5],metrics=['accuracy'])
#




