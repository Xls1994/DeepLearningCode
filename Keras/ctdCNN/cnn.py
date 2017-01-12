'''This example demonstrates the use of Convolution for relation extraction.

multiplayer cnn with different filter size

'''

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,Merge,LSTM,merge,BatchNormalization
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D,GlobalMaxPooling1D
from process_data import make_idx_data_cv
from keras.utils.np_utils import to_categorical
from attention_lstm import AttentionLSTM,AttentionLSTM_t
import  cPickle

from keras import backend as K
np.random.seed(1337)
def CNNmodel():
    nb_filter = 50
    maxlen=10
    filter_sizes =(2,3,4)
    embedding_dim =20
    graph_in = Input(shape=(maxlen, embedding_dim))
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

def loadData(path):
    x = cPickle.load(open(path,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print(len(word_idx_map))
    print(len(vocab))
    print (len(revs))
    datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=116,k=100, filter_h=5)
    img_h = len(datasets[0][0])-1
    print ('img_h',img_h)
    print(datasets[0].shape)
    test_set_x = datasets[1][:,:img_h]
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set_x =datasets[0][:,:img_h]
    train_set_y =np.asarray(datasets[0][:,-1],"int32")
    print (np.shape(train_set_x))
    print('load data...')
    print(np.shape(W))
    print(type(W))
    return (train_set_x,train_set_y),(test_set_x,test_set_y),W

def preditFval(predictions):
        num = len(predictions)
        with open('L_predict_result.txt', 'w') as f:
            for i in range(num):
                if predictions[i][1] > predictions[i][0]:
                    predict = +1
                else:
                    predict = -1
                f.write(str(predict) + str(predictions[i]) + '\n')

        TP = len([1 for i in range(num) if
                  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
        FP = len([1 for i in range(num) if
                  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])
        FN = len([1 for i in range(num) if
                  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
        TN = len([1 for i in range(num) if
                  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])

        print('Wether match? ', (TP + FP + FN + TN) == num)
        print(TP, FP, FN, TN)  # 0 0 1875 9803

        precision = TP / (float)(TP + FP)
        recall = TP / (float)(TP + FN)
        Fscore = (2 * precision * recall) / (precision + recall)  # ZeroDivisionError: integer division or modulo by zero

        print(">> Report the result ...")
        print("-1 --> ", len([1 for i in range(num) if predictions[i][1] < predictions[i][0]]))
        print("+1 --> ", len([1 for i in range(num) if predictions[i][1] > predictions[i][0]]))
        print("TP=", TP, "  FP=", FP, " FN=", FN, " TN=", TN)
        print("precision= ", precision)
        print("recall= ", recall)
        print("Fscore= ", Fscore)

def run():
    np.random.seed(1337)

    # maxlen = 66

    # Convolution
    filter_length = 5
    nb_filter = 50
    pool_length = 4

    # LSTM
    lstm_output_size = 200

    # Training
    batch_size = 30
    nb_epoch = 10

    print('Loading data...')
    import json
    ctdPath ='test.json'
    indexJson = open(ctdPath, "r")
    inputInfo = json.load(indexJson)
    indexJson.close()
    dictPath =inputInfo["ctdEm"]
    dataPath =inputInfo["mrPath"]
    (X_train, y_train), (X_test, y_test), WordEm = loadData(path=dataPath)
    print('datapath:',dataPath)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    # X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    # X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    train_label = to_categorical(y_train, 2)
    test_label = to_categorical(y_test, 2)

    print('set hyper-parameters:')
    max_features = (WordEm.shape[0])
    embedding_size = WordEm.shape[1]
    print('load ctd features...')
    import ctdFeatureData


    ctdWord = np.loadtxt(dictPath, delimiter=' ', dtype='float32')
    train,test = ctdFeatureData.makeidx_map(ctdPath)
    train = np.asarray(train, dtype='int32')
    test = np.asarray(test, dtype='int32')
    print('Build model...')
    maxlen =X_train.shape[1]
    def buildModel():
        from keras.regularizers import l2
        print('xxx')
        main_inputs = Input(shape=(maxlen,), dtype='int32', name='main_input')
        inputs = Embedding(max_features, embedding_size, input_length=maxlen, weights=[WordEm])(main_inputs)
        # x =Dropout(0.25)(inputs)
        convs = []
        filter_sizes = (2, 3, 4)
        for fsz in filter_sizes:
            conv = Convolution1D(nb_filter=nb_filter,
                                 filter_length=fsz,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1,
                                 W_regularizer=l2(l=0.01)
                                 )(inputs)
            pool = MaxPooling1D(pool_length=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)
        out = Merge(mode='concat',concat_axis=1)(convs)
        # out =GlobalMaxPooling1D()(convs)
        out =BatchNormalization()(out)
        # out =LSTM(lstm_output_size,activation='relu')(out)
        predict = Dense(2, activation='softmax',W_regularizer=l2(0.01))(out)
        model = Model(input=main_inputs, output=predict)
        return model

    def buildBiLstm():
        main_inputs = Input(shape=(maxlen,), dtype='int32', name='main_input')
        inputs = Embedding(max_features, embedding_size, input_length=maxlen, weights=[WordEm])(main_inputs)
        lstm1 = LSTM(100)(inputs)
        lstm2 = LSTM(200)(inputs)
        lstm1_back = LSTM(100, go_backwards=True)(inputs)
        # lstm2_back =LSTM(200,go_backwards=True)(inputs)
        out = merge([lstm1, lstm2, lstm1_back], mode='concat')
        out = Dense(200, activation='tanh')(out)
        predict = Dense(2, activation='softmax')(out)
        model = Model(input=main_inputs, output=predict)
        return model

    def buildCNNwithCTD():
        nb_filter = 50
        filter_sizes = (2, 3, 4)
        convs = []
        main_inputs = Input(shape=(maxlen,), dtype='int32', name='main_input')
        inputs = Embedding(max_features, embedding_size, input_length=maxlen, weights=[WordEm])(main_inputs)

        for fsz in filter_sizes:
            conv = Convolution1D(nb_filter=nb_filter,
                                 filter_length=fsz,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1)(inputs)
            pool = MaxPooling1D(pool_length=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)

        if len(filter_sizes) > 1:
            out = Merge(mode='concat')(convs)
        else:
            out = convs[0]
        ctdinput = Input(shape=(1,), dtype='int32', name='ctd_input')
        # ctdword = Embedding(4, 10, input_length=1, weights=[ctdWord])(ctdinput)
        ctdword = Embedding(4, 50, input_length=1)(ctdinput)
        ctdword = Dense(10)(ctdword)
        ctdf = Flatten()(ctdword)
        print(ctdWord.shape)
        outs = merge([out, ctdf], mode='concat')

        predict = Dense(2, activation='softmax')(outs)
        model = Model(input=[main_inputs, ctdinput], output=predict)
        return model
    def attLstm():
        from keras.regularizers import l2
        main_inputs = Input(shape=(maxlen,), dtype='int32', name='main_input')
        inputs = Embedding(max_features, embedding_size, input_length=maxlen, weights=[WordEm])(main_inputs)
        lstm1 = AttentionLSTM_t(100,W_regularizer=l2(0.01))(inputs)
        lstm1_back = AttentionLSTM_t(100, go_backwards=True)(inputs)
        out = merge([lstm1, lstm1_back], mode='concat')
        out = Dense(100, activation='tanh')(out)
        predict = Dense(2, activation='softmax')(out)
        model = Model(input=main_inputs, output=predict)
        return model
    # model =buildCNNwithCTD()
    model = buildModel()
    print('xxxxxx')
    pltname = 'modelcnn-ctd.png'
    savePath = 'result_ctd_score.txt'
    # savePath = 'result_ctd_crossSen.txt'

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    from keras.optimizers import adadelta
    ss = adadelta(clipnorm=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=[precision, 'fbeta_score'])
    model.summary()
    from keras.utils.visualize_util import plot
    plot(model, to_file=pltname)
    print('Train...')

    def trainCTDModel():
        model.fit([X_train, train], train_label, batch_size=batch_size, nb_epoch=2)
        score = model.evaluate([X_test, test], test_label, batch_size=batch_size)
        result = model.predict([X_test, test])
        print(len(score))
        for i in range(len(score)):
            print(score[i])
        # result = model.predict([X_test,test])
        np.savetxt(savePath, result, fmt="%.4f", delimiter=" ")
    def trainModel():
        for i in range(3):
            model.fit([X_train], train_label, batch_size=batch_size, nb_epoch=1,validation_split=0.2,shuffle=True)
            score = model.evaluate([X_test], test_label, batch_size=batch_size)
            result = model.predict([X_test])
            # print(len(score))
            # for i in range(len(score)):
            #     print('xxxx...',score[i])
            np.savetxt('result_'+str(i)+'.txt', result, fmt="%.4f", delimiter=" ")

    trainModel()
    # trainCTDModel()
if __name__=='__main__':
    run()











