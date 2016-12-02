from keras.layers import LSTM
from keras.backend import backend as K

class AttentionLSTM(LSTM):
    def __init__(self,output_dim,attention_vec,**kwargs):
        self.attentionvec =attention_vec
        super(AttentionLSTM, self).__init__(output_dim,**kwargs)
    def build(self, input_shape):
        super(AttentionLSTM,self).build(input_shape)
        assert hasattr(self.attentionvec,'_keras_shape')
        attention_dim =self.attentionvec._keras_shape[1]
        self.U_a =self.inner_init( (self.output_dim,self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a =K.zeros((self.output_dim,),
                          name='{}_b_a'.format(self.name))
        self.U_m = self.inner_init((attention_dim,self.output_dim),
                                   name ='{}_U_m'.format(self.name))
        self.b_m =K.zeros((self.output_dim,),
                          name='{}_b_m'.format(self.name))
        self.U_s = self.inner_init((self.output_dim,self.output_dim),
                                   name ='{}_U_s'.format(self.name))
        self.b_s =K.zeros((self.output_dim,),
                          name='{}_b_s'.format(self.name))
        self.trainable_weights+=[self.U_a,self.U_m,self.U_s,
                                 self.b_a,self.b_m,self.b_s]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del  self.initial_weights
        def step(self, x, states):
            h,[h,c]=super(AttentionLSTM,self).step(x,states)
            attention =states[4]

            m =K.tanh(K.dot(h,self.U_a)+attention+self.b_a)
            s =K.exp(K.dot(m,self.U_s)+self.b_s)
            h = h*s
            return h,[h,c]
        def get_constants(self, x):
            constants =super(AttentionLSTM, self).get_constants(x)
            constants.append (K.dot(self.attentionvec,self.U_m)+self.b_m)
            return constants
def buildModel():
    from keras.layers import Input,Dense,merge
    from keras.models import Model
    import theano.tensor as T
    a =Input(shape=(2,),name='a')
    b =Input(shape=(2,),name='b')
    a_rotated =Dense(2,activation='linear')(a)

    def cosine(x):
        axis =len(x[0]._keras_shape)-1
        dot =lambda a,b: T.tensordot(a,b,axes=axis)
        return dot(x[0],x[1])/T.sqrt(dot(x[0],x[0])* dot(x[1],x[1]))
    cosine_sim =merge([a_rotated,b],mode=cosine,output_shape=lambda x: x[:-1])
    model =Model(input=[a,b],output=[cosine_sim])
    # model =Model(input=[a],output=[a_rotated])
    model.compile(optimizer ='sgd',loss='mse')

    import  numpy as np
    a_data =np.asarray([[0, 1], [1, 0]])
    b_data =np.asarray([[1, 0], [0, -1]])
    targets =np.asarray([1,1,1,1])
    targets =targets.reshape((2,2))
    print a_data.shape
    print targets.shape
    # model.fit([a_data],[b_data],nb_epoch=10)
    model.fit([a_data,b_data],[targets],nb_epoch=10)
    print(model.layers[2].get_value())
    pass

if __name__=='__main__':
    buildModel()