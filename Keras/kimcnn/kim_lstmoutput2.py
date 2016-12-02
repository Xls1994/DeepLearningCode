'''This example demonstrates the use of Convolution for text classification.

multiplayer cnn with different filter size

'''

from __future__ import print_function
import numpy as np
import  cPickle
import  theano.tensor as T
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Merge,LSTM
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D
from process_data import make_idx_data_cv
from keras.utils.np_utils import to_categorical


np.random.seed(1337)


def loadData(path):
    x = cPickle.load(open(path,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print(len(word_idx_map))
    print(len(vocab))
    datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=10,k=100, filter_h=5)
    img_h = len(datasets[0][0])-1
    test_set_x = datasets[1][:,:img_h]
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set_x =datasets[0][:,:img_h]
    train_set_y =np.asarray(datasets[0][:,-1],"int32")
    print (np.shape(train_set_x))
    print('load data...')
    print(np.shape(W))
    print(type(W))
    return (train_set_x,train_set_y),(test_set_x,test_set_y),W

if __name__=='__main__':
    # set  batch parameters:
    max_features = 10650
    maxlen = 18
    embedding_size = 100

    # Convolution
    filter_length = 5
    nb_filter = 64
    pool_length = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30
    nb_epoch = 2
    train,test ,W=loadData('mr_Fscope.p')

    train_set_x=train[0]
    train_set_y =train[1]
    test_set_x,test_set_y =test
    wordEM =[W]
    print ('label type',type(train_set_y))
    def buildModel():
        main_inputs =Input(shape=(maxlen,),dtype='int32',name='main_input')
        inputs =Embedding(max_features, embedding_size, input_length=maxlen,weights=wordEM)(main_inputs)
        # x =Dropout(0.25)(inputs)
        x =Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1)(inputs)
        x =MaxPooling1D(pool_length=pool_length)(x)
        x =LSTM(lstm_output_size)(x)
        predict =Dense(2,activation='softmax')(x)
        model =Model(input=main_inputs,output=predict)
        return model
    model =buildModel()

    model.summary()
    print ('start...')
    trainlabel = to_categorical(train_set_y)
    testlabel =to_categorical(test_set_y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])

    print ('train...on batch...')

    model.fit(train_set_x,trainlabel,32,5)
    result =model.predict(test_set_x)
    cost,acc =model.evaluate(test_set_x,testlabel)
    print ('accuracy',acc)
    np.savetxt('result'+'.txt',result,fmt='%.4f',delimiter=' ')
    json_str =model.to_json()
    open('model.json','w').write(json_str)



    def trainOnbatch(nb_epoch):
        batch_index =0
        batch_size = 50
        for i in range(nb_epoch):
            X_batch =train_set_x[batch_index:batch_size+batch_index,]
            Y_batch =trainlabel[batch_index:batch_size+batch_index,]
            print (np.shape(X_batch))
            cost = model.train_on_batch(X_batch,Y_batch)
            print ('train...cost',cost)
            batch_index += batch_size
            batch_index = 0 if batch_index>=train_set_x.shape[0] else batch_index
            model.save_weights('model'+str(i)+'.hdf5')
    



