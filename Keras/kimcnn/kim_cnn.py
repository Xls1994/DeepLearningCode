'''This example demonstrates the use of Convolution for text classification.

multiplayer cnn with different filter size

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import  theano.tensor as T
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Merge,LSTM,GRU
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D,GlobalMaxPooling1D
from process_data import make_idx_data_cv
from keras.utils.np_utils import to_categorical
import  cPickle
import numpy as np
from keras import backend as K

def CNNmodel():
    nb_filter = 50
    maxlen=10
    filter_sizes =(2,3,4)
    graph_in = Input(shape=(maxlen, embedding_dims))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=nb_filter,
                                 filter_length=fsz,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1)(graph_in)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    return graph
def k_max_pool(self, x, k):
        """
        perform k-max pool on the input along the rows

        input: theano.tensor.tensor4

        k: theano.tensor.iscalar
            the k parameter

        Returns:
        4D tensor
        """
        x = T.reshape(x, (x.shape[0], x.shape[1], 1, x.shape[2] * x.shape[3]))
        ind = T.argsort(x, axis = 3)

        sorted_ind = T.sort(ind[:,:,:, -k:], axis = 3)

        dim0, dim1, dim2, dim3 = sorted_ind.shape

        indices_dim0 = T.arange(dim0).repeat(dim1 * dim2 * dim3)
        indices_dim1 = T.arange(dim1).repeat(dim2 * dim3).reshape((dim1*dim2*dim3, 1)).repeat(dim0, axis=1).T.flatten()
        indices_dim2 = T.arange(dim2).repeat(dim3).reshape((dim2*dim3, 1)).repeat(dim0 * dim1, axis = 1).T.flatten()

        result = x[indices_dim0, indices_dim1, indices_dim2, sorted_ind.flatten()].reshape(sorted_ind.shape)
        shape = (result.shape[0],result.shape[1], result.shape[2] * result.shape[3], 1)

        result = T.reshape(result, shape)

        return result
def kimCNN():
    # Model Hyperparameters
    sequence_length = 56
    embedding_dim = 20
    dropout_prob = (0.25, 0.5)
    vocabulary = 150

    model =Sequential()
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        ))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    model.add(CNNmodel())
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

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

    #set dims
    embedding_dims = 100
    hidden_dims = 50
    nb_epoch =20
    train,test ,W=loadData('mr_Fscope.p')

    train_set_x=train[0]
    train_set_y =train[1]
    test_set_x,test_set_y =test

    print ('label type',type(train_set_y))
    # np.savetxt('label',train_set_y,fmt='%.2f',delimiter=' ')
    wordEM =[W]

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(10650, 100, input_length=18,weights=wordEM))    #(none,18,100)
    model.add(Convolution1D(nb_filter=150,filter_length=3,activation='relu',
                            input_dim=100))
    model.add(MaxPooling1D())
    # We add a vanilla hidden layer:
    model.add(LSTM(50))
    # model.add(Dense(50,activation='relu'))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    model.summary()
    print ('start...')
    trainlabel = to_categorical(train_set_y)
    testlabel =to_categorical(test_set_y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])

    print ('train...on batch...')
    # trainOnbatch(nb_epoch)
    #
    model.fit(train_set_x,trainlabel,32,10)
    result =model.predict(test_set_x)
    cost,acc =model.evaluate(test_set_x,testlabel)
    print ('accuracy',acc)
    np.savetxt('result11'+'.txt',result,fmt='%.2f',delimiter=' ')
    json_str =model.to_json()
    # getlayer2=K.function([model.layers[0].input,K.learning_phase()],[model.layers[4].output])
    # layerout= getlayer2([test_set_x,0])

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
            # testcost,accuracy =model.evaluate(test_set_x, testlabel, verbose=0)
            # print('test cost:',cost,'test accuracy:', accuracy)
            # result =model.predict(test_set_x)
            # np.savetxt('results/result'+str(i)+'.txt',result,fmt='%.2f',delimiter=' ')



