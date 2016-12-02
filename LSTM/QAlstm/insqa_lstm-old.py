############################################################
# lstm + CNN
############################################################

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from loadData import *

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

def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

class LSTM(object):
  def __init__(self, input1, input2, input3, mask1, mask2, mask3, word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters):
    #proj_size means embedding_size
    #'lstm_W' = [embedding_size, embedding_size]
    #'lstm_U' = [embedding_size, embedding_size]
    #'lstm_b' = [embedding_size]
    proj_size = 100 #TODO, what does proj mean
    self.params, tparams = [], {}
    tparams, self.params = param_init_lstm(proj_size, tparams, self.params)
    tparams, self.params = param_init_cnn(filter_sizes, num_filters, proj_size, tparams, self.params)
    lookup_table = theano.shared(word_embeddings, borrow=True)
    tparams['lookup_table'] = lookup_table
    self.params += [lookup_table]

    n_timesteps = input1.shape[0]
    n_samples = input1.shape[1]

    lstm1, lstm_whole1 = self._lstm_net(tparams, input1, sequence_len, batch_size, embedding_size, mask1, proj_size)
    lstm2, lstm_whole2 = self._lstm_net(tparams, input2, sequence_len, batch_size, embedding_size, mask2, proj_size)
    lstm3, lstm_whole3 = self._lstm_net(tparams, input3, sequence_len, batch_size, embedding_size, mask3, proj_size)

    #dimshuffle [sequence_len, batch_size, proj_size] to [batch_size,1, sequence_len, proj_size]
    cnn_input1 = T.reshape(lstm1.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn_input2 = T.reshape(lstm2.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn_input3 = T.reshape(lstm3.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn1 = self._cnn_net(tparams, cnn_input1, batch_size, sequence_len, num_filters, filter_sizes, proj_size)
    cnn2 = self._cnn_net(tparams, cnn_input2, batch_size, sequence_len, num_filters, filter_sizes, proj_size)
    cnn3 = self._cnn_net(tparams, cnn_input3, batch_size, sequence_len, num_filters, filter_sizes, proj_size)

    len1 = T.sqrt(T.sum(cnn1 * cnn1, axis=1))
    len2 = T.sqrt(T.sum(cnn2 * cnn2, axis=1))
    len3 = T.sqrt(T.sum(cnn3 * cnn3, axis=1))

    self.cos12 = T.sum(cnn1 * cnn2, axis=1) / (len1 * len2)
    self.cos13 = T.sum(cnn1 * cnn3, axis=1) / (len1 * len3)

    zero = theano.shared(np.zeros(batch_size, dtype=config.floatX), borrow=True)
    margin = theano.shared(np.full(batch_size, 0.05, dtype=config.floatX), borrow=True)
    diff = T.cast(T.maximum(zero, margin - self.cos12 + self.cos13), dtype=config.floatX)
    self.cost = T.sum(diff, acc_dtype=config.floatX)
    self.accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)

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
    input_matrix = tparams['lookup_table'][T.cast(_input.flatten(), dtype="int32")]
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

    #h means hidden output? c means context? so we'll use h?
    #rval[0] = [sequence_len, batch_size, proj_size], rval[1] the same

    #so preact size must equl to x_(lstm input slice)
    #if you want change lstm h(t) size, 'lstm_U' and 'lstm_b'
    #and precat must be changed to another function, like h*U+b
    #see http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    #f(t) = sigmoid(Wf * [h(t-1),x(t)] + bf)
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

def train():
    batch_size = int(256)
    embedding_size = 100
    learning_rate = 0.05
    n_epochs = 20000000
    validation_freq = 1000
    filter_sizes = [1, 2, 3, 5]
    num_filters = 500

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    trainList = load_train_list()
    testList = load_test_list()
    train_x1, train_x2, train_x3, mask1, mask2, mask3 = load_data(trainList, vocab, batch_size)
    x1, x2, x3 = T.fmatrix('x1'), T.fmatrix('x2'), T.fmatrix('x3')
    m1, m2, m3 = T.fmatrix('m1'), T.fmatrix('m2'), T.fmatrix('m3')
    model = LSTM(
        input1=x1, input2=x2, input3=x3,
        mask1=m1, mask2=m2, mask3=m3,
        word_embeddings=word_embeddings,
        batch_size=batch_size,
        sequence_len=train_x1.shape[0], #row is sequence_len
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters)

    cost, cos12, cos13 = model.cost, model.cos12, model.cos13
    params, accuracy = model.params, model.accuracy
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    p1, p2, p3 = T.fmatrix('p1'), T.fmatrix('p2'), T.fmatrix('p3')
    q1, q2, q3 = T.fmatrix('q1'), T.fmatrix('q2'), T.fmatrix('q3')
    train_model = theano.function(
        [p1, p2, p3, q1, q2, q3], 
        [cost, accuracy], 
        updates=updates,
        givens={
            x1: p1, x2: p2, x3: p3, m1: q1, m2: q2, m3: q3
        }
    )

    v1, v2, v3 = T.matrix('v1'), T.matrix('v2'), T.matrix('v3')
    u1, u2, u3 = T.matrix('u1'), T.matrix('u2'), T.matrix('u3')
    validate_model = theano.function(
        inputs=[v1, v2, v3, u1, u2, u3],
        outputs=[cos12, cos13],
        #updates=updates,
        givens={
            x1: v1, x2: v2, x3: v3, m1: u1, m2: u2, m3: u3
        }
    )

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_x1, train_x2, train_x3, mask1, mask2, mask3 = load_data(trainList, vocab, batch_size)
        #print('train_x1, train_x2, train_x3')
        #print(train_x1.shape, train_x2.shape, train_x3.shape)
        cost_ij, acc = train_model(train_x1, train_x2, train_x3, mask1, mask2, mask3)
        print 'load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij) + ', acc:' + str(acc)
        if epoch % validation_freq == 0:
            print 'Evaluation ......'
            validation(validate_model, testList, vocab, batch_size)

if __name__ == '__main__':
    train()
