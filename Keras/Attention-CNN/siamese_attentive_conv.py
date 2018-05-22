from keras.engine import Input
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, dot, Lambda,RepeatVector
from attention_layer import AttentionLayer
import keras.backend as K

# from keras examples
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# input =  Input(shape=(50,100), dtype='float32')

conv_4 = Conv1D(300,
                4,
                padding='same',
                activation='relu',
                strides=1)
# shared = Model(input, conv_4)

input_1 =  Input(shape=( 20,100), dtype='float32')
input_2 =  Input(shape=( 15,100), dtype='float32')

out_1 = conv_4(input_1)
out_2 = conv_4(input_2)
print 'out1 shape...',K.int_shape(out_1)
print 'out2 shape...',K.int_shape(out_2)
attention = AttentionLayer()([out_1,out_2])

# out_1 column wise
att_1 = GlobalMaxPooling1D()(attention)
att_1 = Activation('softmax')(att_1)
print 'attention shape',K.int_shape(att_1)
att_1 =Lambda(lambda x: K.expand_dims(x, 2))(att_1)
out1 = dot([att_1, out_2], axes=1)


# out_2 row wise
attention_transposed = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(attention)
att_2 = GlobalMaxPooling1D()(attention_transposed)
att_2 = Activation('softmax')(att_2)
att_2 =Lambda(lambda x: K.expand_dims(x, 2))(att_2)
out2 = dot([att_2, out_1], axes=1)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([out1, out2])

model = Model(inputs=[input_1, input_2], outputs=distance)
from keras.utils.vis_utils import plot_model
plot_model(model,'model.png',show_shapes=True)
