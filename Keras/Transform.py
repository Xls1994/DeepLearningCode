
import theano.tensor as T

from keras.layers.core import Layer
import numpy as np

class Transform(Layer):
     '''
         This function is needed transform the dataset such that we can
         add mlp layers before RNN/LSTM or after the RNN/LSTM.
     '''
     def __init__(self, dims, input=None):
         '''
         If input is three dimensional tensor3, with dimensions (nb_samples, sequence_length, vector_length)
         and dims is tuple (vector_length,), then the output will be (nb_samples * sequence_length, vector_length)
         If input is two dimensional matrix, with dimensions (nb_samples * sequence_length, vector_length)
         and if we want to revert back to (nb_samples, sequence_length, vector_length) so that we can feed
         the LSTM, then we can set dims as (sequence_length, vector_length).
         This function is needed for adding mlp layers before LSTM or after the LSTM.
         When used as first layer, input has to be set either as tensor3 or matrix
         '''

         super(Transform, self).__init__()
         self.dims = dims
         if input is not None:
             self.input = input

     def get_output(self, train):
         X = self.get_input(train)
         first_dim = T.prod(X.shape) / np.prod(self.dims)
         return T.reshape(X, (first_dim,)+self.dims)

     def get_config(self):
         return {"name":self.__class__.__name__,
             "dims":self.dims}