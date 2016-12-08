############################################################
# lstm + CNN
############################################################
import theano
import theano.tensor as T
import  sys
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from loadData import *
from collections import OrderedDict
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def param_init_cnn(filter_sizes, num_filters, proj_size, tparams, grad_params):
    rng = np.random.RandomState(23455)
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, proj_size)
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        tparams['cnn_W_' + str(filter_size)] = W
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        tparams['cnn_b_' + str(filter_size)] = b
        grad_params += [W, b]
    return tparams, grad_params

def param_init_lstm(proj_size, tparams, grad_params):
    W = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    W_t = theano.shared(W, borrow=True)
    tparams[_p('lstm', 'W')] = W_t
    U = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    U_t = theano.shared(U, borrow=True)
    tparams[_p('lstm', 'U')] = U_t
    b = np.zeros((4 * proj_size,))
    b_t = theano.shared(b.astype(config.floatX), borrow=True)
    tparams[_p('lstm', 'b')] = b_t
    grad_params += [W_t, U_t, b_t]

    return tparams, grad_params

def param_init_lr(insize,outsize,tparams,grad_params):
    W =np.zeros((insize,outsize),dtype=config.floatX)
    b = np.zeros((outsize,),dtype=config.floatX)
    W_t =theano.shared(value=W,borrow=True)
    b_t =theano.shared(value=b,borrow=True)
    tparams[_p('lr','W')]=W_t
    tparams[_p('lr','b')]=b_t
    grad_params +=[W_t,b_t]
    return tparams, grad_params

def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj
class LSTM_Lr (object):
    def __init__(self, input1, mask1,label, word_embeddings, batch_size, sequence_len, embedding_size,filter_sizes, num_filters):
        proj_size =100
        in_size =num_filters * len(filter_sizes)
        out_size =2
        self.params,tparams =[],{}
        tparams,self.params =param_init_lstm(proj_size,tparams,self.params)
        tparams,self.params =param_init_cnn(filter_sizes,num_filters,proj_size,tparams,self.params)
        tparams,self.params =param_init_lr(in_size,out_size,tparams,self.params)
        look_table =theano.shared(word_embeddings, borrow= True)
        tparams['look_table'] =look_table
        self.params +=[look_table]

        n_timesteps =input1.shape[0]
        n_samples =input1.shape[1]
        lstm1, lstm_whole1 = self._lstm_net(tparams, input1, sequence_len, batch_size, embedding_size, mask1, proj_size)
        cnn_input1 = T.reshape(lstm1.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
        cnn1 = self._cnn_net(tparams, cnn_input1, batch_size, sequence_len, num_filters, filter_sizes, proj_size)

        p_y_given_x,y_perd =self._lr_net(tparams,cnn1,in_size,out_size)
        self.p_y_given_x =p_y_given_x
        self.y_pred =y_perd
        off =1e-8
        self.cost = -T.log(self.p_y_given_x[T.arange(batch_size), label] + off).mean()
        self.f_pred = theano.function([input1, mask1], y_perd, name='f_pred')
        self.f_pred_prob =theano.function([input1,mask1],p_y_given_x,name='f_pred_prob')
        self.errors =T.mean(T.neq(self.y_pred, label))
        self.tparams =tparams

    def _lr_net(self,tparams,lr_input,in_size,outsize):
        W =tparams['lr_W']
        b =tparams['lr_b']
        p_y_given_x = T.nnet.softmax(T.dot(lr_input, W) + b)
        y_pred = T.argmax(p_y_given_x, axis=1)

        return p_y_given_x,y_pred

    def _cnn_net(self, tparams, cnn_input, batch_size, sequence_len, num_filters, filter_sizes, proj_size):
        outputs = []
        for filter_size in filter_sizes:
            filter_shape = (num_filters, 1, filter_size, proj_size)
            image_shape = (batch_size, 1, sequence_len, proj_size)
            W = tparams['cnn_W_' + str(filter_size)]
            b = tparams['cnn_b_' + str(filter_size)]
            conv_out = conv2d(input=cnn_input, filters=W, filter_shape=filter_shape, input_shape=image_shape)
            pooled_out = pool.pool_2d(input=conv_out, ds=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
            pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
            outputs.append(pooled_active)
        num_filters_total = num_filters * len(filter_sizes)
        output_tensor = T.reshape(T.concatenate(outputs, axis=1), [batch_size, num_filters_total])
        return output_tensor
    def _lstm_net(self, tparams, _input, sequence_len, batch_size, embedding_size, mask, proj_size):

        input_matrix = tparams['look_table'][T.cast(_input.flatten(), dtype="int32")]
        input_x = input_matrix.reshape((sequence_len, batch_size, embedding_size))
        proj, proj_whole = lstm_layer(tparams, input_x, proj_size, prefix='lstm', mask=mask)
        #if useMask == True:
        #proj = (proj * mask[:, :, None]).sum(axis=0)
        #proj = proj / mask.sum(axis=0)[:, None]
        #if options['use_dropout']:
        #proj = dropout_layer(proj, use_noise, trng)
        return proj, proj_whole
#state_below is word_embbeding tensor(3dim)
def lstm_layer(tparams, state_below, proj_size, prefix='lstm', mask=None):
    #dim-0 steps, dim-1 samples(batch_size), dim-3 word_embedding
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1
    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, proj_size))
        f = T.nnet.sigmoid(_slice(preact, 1, proj_size))
        o = T.nnet.sigmoid(_slice(preact, 2, proj_size))
        c = T.tanh(_slice(preact, 3, proj_size))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        #if mask(t-1)==0, than make h(t) = h(t-1)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = proj_size
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0], rval[1]

def _p(pp, name):
    return '%s_%s' % (pp, name)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of wfhat is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def adadelta(lr, tparams, grads, x_zheng, x_zheng_mask, y, cost):
    """
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x_zheng, x_zheng_mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared',allow_input_downcast=True)

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared,  f_update

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, test_index in iterator:
        x_zheng, x_zheng_mask, y = prepare_data([data[0][t] for t in test_index],
                                  np.array(data[1])[test_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x_zheng, x_zheng_mask)
        probs[test_index, :] = pred_probs

        n_done += len(test_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x_zheng, x_zheng_mask, y = prepare_data([data[0][t] for t in valid_index],
                                   np.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x_zheng, x_zheng_mask)
        targets = np.array(data[1])[valid_index]

        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def get_proj(f_proj, prepare_data, data, iterator, dim_proj, verbose=False):
    """
    Get the top hidden layer
    """
    n_samples = len(data[0])
    projs = np.zeros((n_samples, 2*dim_proj)).astype(config.floatX)

    for _, index in iterator:
        x_zheng, x_zheng_mask, x_ni, x_ni_mask, y = prepare_data([data[0][t] for t in index],
                                  [data[1][t] for t in index],
                                  np.array(data[2])[index],
                                  maxlen=None)
        hidden_projs = f_proj(x_zheng, x_zheng_mask, x_ni, x_ni_mask)
        projs[index, :] = hidden_projs

    return projs

def trainLstm():
    import time
    validFreq=100,  # Compute the validation error after this number of update.
    saveFreq=200,  # Save the parameters after every saveFreq updates
    path ='QAcorpus/Word/mr_FscopeContexts.pkl'
    dispFreq=10
    test_size =11954
    test_batch_size = int(256)
    embedding_size = 100
    n_epochs = 20
    filter_sizes = [1, 2, 3]
    num_filters = 500
    maxLen =10
    lrate=0.01
    saveto='1.npz'
    trainSet,testSet,word_embeddings = load_data(path,n_words=10733,maxlen=10)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(testSet[0]))
        # numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([testSet[0][n] for n in idx], [testSet[1][n] for n in idx])


    print 'Building model'
    x = T.matrix('x')
    m= T.fmatrix('m')
    y =T.ivector('y')
    model =LSTM_Lr(input1=x,mask1=m,label= y,word_embeddings=word_embeddings,
                   batch_size=test_batch_size,
                   sequence_len=maxLen,
                   embedding_size=embedding_size,
                   filter_sizes=filter_sizes,
                   num_filters=num_filters

    )
    cost =model.cost
    # f_cost = theano.function([x,m, y], cost, name='f_cost')
    tparams, errors = model.tparams, model.errors
    grads = T.grad(cost, wrt=tparams.values())
    # f_grad = theano.function([x,m, y], grads, name='f_grad')
    f_pred =model.f_pred
    f_pred_prob =model.f_pred_prob
    lr = T.scalar(name='lr')
    f_grad_shared,   f_update \
        = adadelta(lr, tparams, grads, x, m,  y, cost)

    print 'Optimization'

    kf_test = get_minibatches_idx(len(testSet[0]), test_batch_size)

    print "%d train examples" % len(trainSet[0])
    print "%d test examples" % len(testSet[0])

    history_errs = []
    best_p = None
    bad_count = 0
    if validFreq == -1:
        validFreq = len(trainSet[0]) / test_batch_size
    if saveFreq == -1:
        saveFreq = len(trainSet[0]) / test_batch_size

    uidx = 0  # the number of update done

    kf_train_sorted = get_minibatches_idx(len(trainSet[0]), test_batch_size)

    estop = False  # early stop
    start_time = time.time()

    try:
        updateTime = 0

        f = open('/record.txt', 'w')

        for eidx in xrange(n_epochs):
            n_samples = 0
            kf = get_minibatches_idx(len(trainSet[0]), test_batch_size, shuffle=False)
            for _, train_index in kf:
                uidx += 1
                # use_noise.set_value(1.)

                y = [trainSet[1][t]for t in train_index]
                x_zheng = [trainSet[0][t]for t in train_index]
                x1, x1_mask, y = prepare_data(x_zheng, y)
                n_samples += x1.shape[1]
                print  n_samples
                print x1.shape

                cost = f_grad_shared(x1, x1_mask, y)
                f_update(lrate)
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if np.mod(uidx, validFreq) == 0:
                    # use_noise.set_value(0.)

                    train_err = pred_error(f_pred, prepare_data, trainSet, kf)
                    print('22222222222')
                    test_err = pred_error(f_pred, prepare_data, testSet, kf_test)
                    print('33333333333' )

                    history_errs.append([test_err])
                    print('4444444444' )
                    print('Accuracy:' + str(1-float(test_err)) )
                    f.write('Accuracy:' + str(1-float(test_err)))
                    f.write('\n')

                    if (uidx == 0 or
                        test_err <= np.array(history_errs)[:].min()):

                        best_p = unzip(model.tparams)
                        bad_counter = 0
                        test_prob_best = pred_probs(f_pred_prob, prepare_data, test, kf_test)

                        np.savetxt('QAcorpus/test_best.txt', test_prob_best, fmt='%.4f', delimiter=' ')


                    print ('Train ', train_err, 'Test ', test_err)

            test_prob = pred_probs(f_pred_prob, prepare_data, test, kf_test)

            np.savetxt('QAcorpus/test_prob_'+str(eidx)+'.txt', test_prob, fmt='%.4f', delimiter=' ')

            print 'Seen %d samples' % n_samples

            if estop:
                break

        f.close()
    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    train_err = pred_error(f_pred, prepare_data, trainSet, kf_train_sorted)
    test_err = pred_error(f_pred, prepare_data, testSet, kf_test)

    print 'Train ', train_err, 'Test ', test_err
    if saveto:
        np.savez(saveto, train_err=train_err, test_err=test_err,
                    history_errs=history_errs, **best_p)

    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, test_err

    pass

def train():
    import cPickle
    path ='QAcorpus/Word/mr_FscopeContexts.pkl'
    batch_size = int(256)
    embedding_size = 100
    n_epochs = 20
    filter_sizes = [1, 2, 3]
    num_filters = 500
    maxLen =10
    lrate=0.01
    saveto='1.npz'
    print('load data')
    f =open(path,'r')
    trainData = cPickle.load(f)
    testSet = cPickle.load(f)
    embeddings =cPickle.load(f)
    f.close()
    print('select random dataset')
    trainSet =selectRandomDataSet(batch_size=batch_size,train_x=trainData[0],train_y=trainData[1])
    word_embeddings =embeddings[0]
    print 'Building model'
    x = T.imatrix('x')
    m= T.fmatrix('m')
    y =T.ivector('y')
    model =LSTM_Lr(input1=x,mask1=m,label= y,word_embeddings=word_embeddings[1:,:],
                   batch_size=batch_size,
                   sequence_len=maxLen,
                   embedding_size=embedding_size,
                   filter_sizes=filter_sizes,
                   num_filters=num_filters

    )
    cost =model.cost
    tparams, errors = model.tparams, model.errors
    grads = T.grad(cost, wrt=tparams.values())
    f_pred =model.f_pred
    f_pred_prob =model.f_pred_prob
    lr = T.scalar(name='lr')
    f_grad_shared,   f_update \
        = adadelta(lr, tparams, grads, x, m,  y, cost)

    print 'Optimization'
    print "%d train examples" % len(trainSet[0])
    print 'type....',type(trainSet)
    print 'type....', type(trainSet[0])
    print 'shape...', np.shape(trainSet[0])

    print 'shape ...label',np.shape(trainSet[1])

    epoch =0
    uidx =0
    for epoch in range(n_epochs):
        kf = get_minibatches_idx(len(trainSet[0]), batch_size, shuffle=False)
        # print kf
        for _, train_index in kf:
            uidx +=1
            if len(train_index)!=batch_size:
                print 'error'
                break
            y = [trainSet[1][t] for t in train_index]
            print('type y....',y)
            x = [trainSet[0][t]for t in train_index]
            x, mask, y = prepare_data(x, y)
            cost =f_grad_shared(x,mask,y)
            f_update(lrate)
            print 'update',uidx,'cost:',cost
        train_err = pred_error(f_pred, prepare_data, trainSet, kf)
        print 'train err',train_err



if __name__ == '__main__':
    train()
