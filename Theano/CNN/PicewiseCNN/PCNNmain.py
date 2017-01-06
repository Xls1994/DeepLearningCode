import theano
import theano.tensor as T
import numpy as np
import time
from convLayer import LeNetPositionPoolLayer,MLPDropout,Sigmoid,Tanh
from conv_sentence import sgd_updates_adadelta
#PiceWise CNN model for entitiy classification

def shared_dataset(data_xy, borrow=True):

    data_x, data_y,data_z = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                            dtype='int32'),
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
    shared_z = theano.shared(np.asarray(data_z,
                                            dtype='int32'),
                                 borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32'), shared_z
def train_conv_net(datasets,
                    U,
                    PF1,
                    PF2,
                    filter_hs=3,
                    conv_non_linear="tanh",
                    hidden_units=[100, 50],
                    shuffle_batch=True,
                    epochs=25,
                    sqr_norm_lim=9,
                    lr_decay=0.95,
                    static=False,
                    batch_size=1,
                    img_w=10,
                    pf_dim=5,
                    norm=0,
                    dropout_rate=[0.5],
                    directory='./',
                    activations_str=[],
                    borrow=True):
    # T.config.exception_verbosity='high'
    activations = []
    for act in activations_str:
        dropout_rate.append(0.5)
        if act.lower() == 'tanh':
            activations.append(Tanh)
        elif act.lower() == 'sigmoid':
            activations.append(Sigmoid)

    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0]) - 3
    filter_w = img_w # img_w = 50

    feature_maps = hidden_units[0]
    # filter_shape = (feature_maps, 1, filter_hs, filter_w+pf_dim*2)
    filter_shape=(feature_maps,1,filter_hs,filter_w)
    n_epochs=10
    x = T.imatrix('x')
    index = T.lscalar()
    # p1 = T.imatrix('pf1')
    # p2 = T.imatrix('pf2')
    pool_size = T.imatrix('pos')
    y = T.ivector('y')

    Words = theano.shared(value=U, name="Words")
    # PF1W = theano.shared(value=PF1, name="pf1w")
    # PF2W = theano.shared(value=PF2, name="pf2w")

    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w,dtype='float32')
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))])

    # zero_vec_tensor = T.vector()
    # zero_vec_pf = np.zeros(pf_dim,dtype='float32')
    # set_zero_pf1 = theano.function([zero_vec_tensor], updates=[(PF1W, T.set_subtensor(PF1W[0,:], zero_vec_tensor))])
    # set_zero_pf2 = theano.function([zero_vec_tensor], updates=[(PF2W, T.set_subtensor(PF2W[0,:], zero_vec_tensor))])

    # The first input layer
    # All the input tokens in a sentence are firstly transformed into vectors by looking up word embeddings.
    # input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    # input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
    # input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

    # layer0_input = T.concatenate([input_words, input_pf1, input_pf2], axis=3)
    layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))

    conv_layer = LeNetPositionPoolLayer(rng, input=layer0_input,
                                    image_shape=(batch_size, 1, img_h, img_w),
                                    filter_shape=filter_shape, pool_size=pool_size,
                                    non_linear=conv_non_linear, max_window_len=3)
    layer1_input = conv_layer.output.flatten(2)

    # the number of hidden unit 0 equals to the features multiple the number of filter (100*1=100)
    hidden_units[0] = feature_maps*3
    classifier = MLPDropout(rng, input=layer1_input,
                            layer_sizes=hidden_units,
                            activations=activations,
                            dropout_rates=dropout_rate)
    params = classifier.params # sofmax parameters
    params += conv_layer.params # conv parameters

    if not static:  # if word vectors are allowed to change, add them as model parameters
        params += [Words]
        # params += [PF1W]
        # params += [PF2W]
    print params
    print type(params)
    for p in params:
        print type(p)
    model_static = [(batch_size, 1, img_h, img_w+pf_dim*2), filter_shape, conv_non_linear, pool_size]
    model_static += [rng, hidden_units, activations, dropout_rate]

    p_y_given_x = classifier.p_y_given_x
    cost =classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    #train data split
    #shuffle train dataset and assign to mini batches.
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0] / batch_size
    n_train_batches = int(np.round(n_batches * 0.9))
    print 'train batch',n_train_batches
    test_set_x = datasets[1][:,:img_h]
    test_set_y = np.asarray(datasets[1][:,-3],"int32")
    test_pool =np.asarray(datasets[1][:,-2:],'int32')
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]
    train_set_x, train_set_y,train_pool = shared_dataset((train_set[:,:img_h],train_set[:,-3],train_set[:,-2:]))
    val_set_x, val_set_y,val_pool = shared_dataset((val_set[:,:img_h],val_set[:,-3],val_set[:,-2:]))
    n_val_batches = n_batches - n_train_batches
    print 'train...set',(train_set[:,:img_h])
    print 'label...',train_set[:,-3]
    print 'pool...',train_set[:,-2:]

    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    # test_input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    # test_input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
    # test_input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

    # test_layer0_input = T.concatenate([test_input_words, test_input_pf1, test_input_pf2], axis=3)

    test_layer0_output = conv_layer.predict(test_layer0_input, test_size, pool_size)
    test_layer1_input = test_layer0_output.flatten(2)
    p_y_given_x = classifier.predict_p(test_layer1_input)
    test_y_pred =classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))


    val_model = theano.function([index], classifier.errors(y),
                                givens={
                                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                    pool_size: val_pool[index * batch_size: (index + 1) * batch_size],
                                    y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)

    # compile theano functions to get train/val/test errors

    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      pool_size: train_pool[index * batch_size: (index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]},
                                  allow_input_downcast=True)
    #start training over mini-batches
    test_model_all = theano.function([x, pool_size, y], test_error, allow_input_downcast=True)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print '... training start at  ' + str(now)

    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    cost_epoch = 0
    f =open('alfa.txt','w')
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):

                a= test_set_x
                b=test_pool
                c= test_set_y
                print 'a',a
                print 'b',b
                print  'c', c
                print np.shape(c)
                print type(c)
                tt =test_model_all(a,b,c)

                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)

        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)

        train_losses = [train_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)

        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        test_loss = test_model_all(test_set_x,test_pool,test_set_y)
        test_perf = 1- test_loss
        print("test perf: %.2f" %(test_perf * 100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x,test_pool,test_set_y)
            test_perf = 1- test_loss
            # np.savetxt( "/pred_best.txt", pred, fmt='%.5f', delimiter=' ');


        # np.savetxt(outputPath + "/hiddenlayer_" + str(epoch) + ".txt", alfaOutputNum, fmt='%.5f', delimiter=' ');

        # np.savetxt(outputPath + "/hiddenlayer2_" + str(epoch) + ".txt", testhidden2, fmt='%.5f', delimiter=' ');
        f.close()
    return test_perf


if __name__=="__main__":
    # pass
    train =np.loadtxt('npdata',dtype='int32',delimiter=' ')
    em =np.loadtxt('embeddings.txt',dtype='float32',delimiter=' ')
    # print train
    # print em
    datasets =[train,train]

    n_batches =10
    n_train_batches = int(np.round(n_batches * 0.9))
    print 'n-train',n_train_batches
    ff =train_conv_net(datasets,em,PF1=1,PF2=2,img_w=10,batch_size=2)
    # test_set_x = datasets[1][:, :10]
    # test_set_y = np.asarray(datasets[1][:, -3], "int32")
    # test_pool = np.asarray(datasets[1][:, -2:], 'int32')
    # print 'pool',test_pool
    # index =1
    # batch_size =1
    # x =test_pool[index * batch_size: (index + 1) * batch_size]
    # print 'xxx',x
    # for minibatch_index in np.random.permutation(range(n_train_batches)):
    #     # cost_epoch = train_model(minibatch_index)
    #     print minibatch_index