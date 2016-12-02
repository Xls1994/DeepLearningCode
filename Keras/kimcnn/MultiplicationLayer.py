from keras.engine import Layer
from keras import initializations
class MultiplicatonLayer(Layer):
    def __init__(self,**kwargs):
        self.init = initializations.get('glorot_uniform')
        super(MultiplicatonLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        assert  len(input_shape) ==2 and input_shape[1]==1
        self.multiplicand =self.init(input_shape[1: ],name ='multiplicand')
        self.trainable_weights =[self.multiplicand]
    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape)==2 and input_shape[1]==1
        return  input_shape
    def call(self, x, mask=None):
        return x *self.multiplicand

from keras.layers import Input
from keras.models import  Model

inputs =Input(shape=(1,),dtype='int32')
multiply =MultiplicatonLayer()(inputs)
model =Model(input=[inputs],output=multiply)
model.compile(optimizer ='sgd',loss='mse')

import numpy as np
input_data =np.arange(10)
output_data =3*input_data

model.fit([input_data],[output_data],nb_epoch=10)
print (model.layers[1].multiplicand.get_value())

