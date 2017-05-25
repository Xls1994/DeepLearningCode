__author__ = 'yangyl'
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import  SGD
from sklearn import  preprocessing
from keras.utils.vis_utils import plot_model
import  numpy as np
import  matplotlib.pyplot as plt
#train data
x_train =np.linspace(-2*np.pi,2*np.pi,1000)
x_train =np.array(x_train).reshape((len(x_train),1))
n =0.1*np.random.rand(len(x_train),1)
y_train =np.sin(x_train)+n

x_train =preprocessing.scale(x_train)
scaler =preprocessing.StandardScaler().fit(x_train)
y_train=scaler.transform(y_train)

#test data
x_test =np.linspace(-5,5,200)
x_test=np.array(x_test).reshape((len(x_test),1))
y_test =np.sin(x_test)

x_test =scaler.transform(x_test)

#prediction data
x_prd =np.linspace(-3,3,101)
x_prd =np.array(x_prd).reshape((len(x_prd),1))
x_prd =scaler.transform(x_prd)
y_prd =np.sin(x_prd)
#plot testing data
fig,ax =plt.subplots()
ax.plot(x_prd,y_prd,'r')

model =Sequential()
model.add(Dense(100,input_dim=1))
model.add(Activation('relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='tanh'))
model.summary()
model.compile(loss='mean_squared_error',optimizer=SGD()
,metrics=['accuracy'])
hist =model.fit(x_test,y_test,batch_size=10,nb_epoch=20)
out =model.predict(x_prd,batch_size=1)
ax.plot(x_prd,out,'k--',lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
plot_model(model,'model.png')
