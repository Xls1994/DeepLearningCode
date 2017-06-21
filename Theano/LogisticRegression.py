import theano
import  theano.tensor as T
import  numpy as np
import gzip
import cPickle
import timeit
class NN(object):
    def __init__(self,inputs,n_in,n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.params =[self.W,self.b]
        self.input=inputs
        self.p_y_given_x =T.nnet.softmax(T.dot(inputs,self.W)+self.b)
        self.y_pred_x = T.argmax(self.p_y_given_x,axis=1)
    def get_p_y_givenx(self):
        return self.p_y_given_x
    def cost(self, y):

        return -T.mean(T.log(self.p_y_given_x[T.arange(y.shape[0]),y]))
    def error(self,y):
        prediction =self.y_pred_x
        return T.mean(T.neq(prediction,y))
def load_data(dataSet):
    f =gzip.open(dataSet)
    train,valid,test=cPickle.load(f)
    def share_data(xy):
        x,y =xy
        x =theano.shared(np.asarray(x,theano.config.floatX))
        y =theano.shared(np.asarray(y,theano.config.floatX))
        return [x,T.cast(y,'int32')]
    trainx,trainy =share_data(train)
    vaildx,vaildy =share_data(valid)
    testx,testy =share_data(test)
    return [(trainx,trainy),(vaildx,vaildy),(testx,testy)]
def train(n_epoches):
    model_options =locals().copy()
    print 'model options',model_options
    dataSetPath ='mnist.pkl.gz'
    batch_size=100
    rng =np.random.RandomState(1123)
    datasets =load_data(dataSetPath)
    train_x,train_y = datasets[0]
    test_x,test_y = datasets[2]
    n_train_batches = train_x.get_value(borrow=True).shape[0]
    print(train_x.get_value(borrow=True).shape[0])
    print (train_x.get_value(borrow=True).shape)
    n_test_batches = test_x.get_value(borrow=True).shape[0]
    print(test_x.get_value(borrow=True).shape[0])
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    x =T.matrix('x',dtype=theano.config.floatX)
    y =T.ivector('y')
    index =T.lscalar()
    layerinput = x.reshape((batch_size,28*28))
    lr = NN(x,n_in=28 * 28,n_out=10)
    params_model =lr.params
    cost =lr.cost(y)
    learning_rate=0.01
    test_model =theano.function([index],lr.error(y),
                                givens={
                                    x:test_x[index*batch_size:(index+1)*batch_size],
                                    y:test_y[index*batch_size:(index+1)*batch_size]
                                })
    grads =T.grad(cost,params_model)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params_model, grads)
    ]
    train_model =theano.function([index],cost,
                                 givens={
                                     x:train_x[index*batch_size:(index+1)*batch_size],
                                    y:train_y[index*batch_size:(index+1)*batch_size]
                                 })
    print 'training.....'
    patience =1000
    patience_increase =2
    improvement_threshold =0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss =np.inf
    best_iter =0
    test_score =0.
    start_time =timeit.default_timer()
    epoch =0
    while(epoch<n_epoches):
        epoch=epoch+1
        for minibatchIndex in xrange(n_train_batches):
            iters =(epoch-1)*n_train_batches +minibatchIndex
            if iters%100 ==0:
                print 'traing @iter',iters
            costij=train_model(minibatchIndex)
            if (iters+1)%validation_frequency==0:
                testloss =[test_model(i) for i in xrange(n_test_batches)]
                this_validation_loss =np.mean(testloss)
                print ('epoch %i, minibatch %i/%i, cost %f test error %f %%' %
                      (epoch, minibatchIndex + 1, n_train_batches, costij,
                       this_validation_loss * 100.))
                if this_validation_loss<best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iters * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iters

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, cost %f test error of '
                           'best model %f %%') %
                          (epoch, minibatchIndex + 1, n_test_batches, costij,
                           test_score * 100.))




if __name__=='__main__':
    train(n_epoches=10)

    # f =gzip.open('mnist.pkl.gz')
    # train=cPickle.load(f)
    # print len(train)