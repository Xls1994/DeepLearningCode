__author__ = 'yangyl'
#stacked LSTM model for predict airline passengers problem with memory
import numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt
import math
np.random.seed(8)
def create_dataset(data,look_back=1):
    #look_back windowsize
    dataX,dataY=[],[]
    for i in range(len(data)-look_back-1):
        n =data[i:(i+look_back),0]
        dataX.append(n)
        dataY.append(data[i+look_back,0])
    return np.array(dataX),np.array(dataY)
if __name__=='__main__':
    print 'test'
    datasets =pd.read_csv('international-airline-passengers.csv',usecols=[1], skipfooter=3,engine='python')
    # plt.plot(datasets)
    # plt.show()
    from keras.models import Sequential
    from keras.layers import Dense,LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    dataset=datasets.values
    dataset=dataset.astype('float32')
    #normalize the dataset
    scaler =MinMaxScaler(feature_range=(0,1))
    dataset=scaler.fit_transform(dataset)
    #split into train and test sets
    train_size =int(len(dataset)*0.67)
    test_size =len(dataset)-train_size
    train,test =dataset[0:train_size,:],dataset[train_size:,:]
    print len(train),len(test)
    #create data
    look_back =3
    trainX,trainY =create_dataset(train,look_back)
    testX,testY =create_dataset(test,look_back)

    #reshape the data [sample,features] into [sample,time-steps,features]
    trainX =np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
    testX =np.reshape(testX,(testX.shape[0],testX.shape[1],1))

    #build LSTM model
    batch_size =1
    model=Sequential()
    model.add(LSTM(4,batch_input_shape=(batch_size,look_back,1),stateful=True,
                   return_sequences=True))
    model.add(LSTM(4,batch_input_shape=(batch_size,look_back,1),
                   stateful=True))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mean_squared_error',optimizer='adam')
    for i in range(100):
        model.fit(trainX,trainY,nb_epoch=100,batch_size=1,verbose=2)
        model.reset_states()
    #predict
    trainPre =model.predict(trainX,batch_size=batch_size)
    model.reset_states()
    testPre =model.predict(testX,batch_size=batch_size)
    #invert predictions
    trainPredict = scaler.inverse_transform(trainPre)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPre)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

