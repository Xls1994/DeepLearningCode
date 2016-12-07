import theano
import theano.tensor as T
import numpy as np

if __name__=='__main__':
    x =np.asarray([[1,2],[1,2]],dtype='float32')
    lens =x.shape[0]
    y =np.zeros(lens,dtype='float32')
    z =np.full(lens, 1, dtype='float32')
    ll =np.asarray([1,0],dtype='int32')
    lll =np.asarray([[1,2,5],[2,3,4]],dtype='float32')


    zero =T.vector('zero',dtype='float32')
    margin =T.vector('margin',dtype='float32')
    cos12 =T.matrix(dtype='float32')
    label =T.vector(dtype='int32')
    # T.reshape(label,(label.shape[0],1))

    diff = T.cast(T.maximum(zero, margin - cos12[:,label]), dtype='float32')


    cost = T.sum(diff, acc_dtype='float32')

    f =theano.function([cos12,zero,margin,label],diff)

    print  f(x,y,z,ll)

    print x[:,0]