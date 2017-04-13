import os
from PIL import Image
import tensorflow as tf
import tensorlayer as tl
import traceback
import Data_Convert as dc
#from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v1 import inception_v1,inception_v1_arg_scope
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import random_ops
import time
slim = tf.contrib.slim

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes=["InceptionV1/Logits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    checkpoints_dir = "D:\\zero\\work\\models-master\\model\\"
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
      os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
      variables_to_restore)

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
net_in = tl.layers.InputLayer(x, name='input_layer')
with slim.arg_scope(inception_v1_arg_scope()):
    network = tl.layers.SlimNetsLayer(layer=net_in, slim_layer=inception_v1,
                                    slim_args= {
                                            'num_classes':19,
                                            'is_training':True,
                                            'dropout_keep_prob':0.5,
                                            'prediction_fn':slim.softmax,
                                            'spatial_squeeze':True,
                                            'reuse':None,
                                            'scope':'InceptionV1'
                                            },
                                        name='InceptionV1'  # <-- the name should be the same with the ckpt model
                                        )

y = network.outputs
probs = tf.nn.softmax(y)
cost = tl.cost.cross_entropy(y, y_, name='cost')
correct = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct, tf.float32))                                              
learning_rate = 0.001
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost,var_list = network.all_params) 
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
network.print_params()
# variables_to_restore = slim.get_variables_to_restore()#(exclude=['InceptionV1/Logits'])
# saver = tf.train.Saver(variables_to_restore)  # init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
# data = saver.restore(sess,"model/inception_v1.ckpt") # init_fn(sess)
init_fn = get_init_fn()
init_fn(sess)

batch_size = 8
train_fileName = "vehicle_train.tfrecords"
test_fileName = "vehicle_test.tfrecords"
x_train,y1_train,y2_train = dc.read_and_decode(train_fileName) 
x_test,y1_test,y2_test = dc.read_and_decode(test_fileName)
x_train_batch, y1_train_batch,y2_train_batch = tf.train.shuffle_batch([x_train,y1_train,y2_train],
                                                    batch_size=batch_size,
                                                    capacity=64,
                                                    min_after_dequeue=16,
                                                    num_threads=4) # set the number of threads here

x_test_batch, y1_test_batch,y2_test_batch = tf.train.batch([x_test,y1_test,y2_test],
                                                    batch_size=batch_size,
                                                    capacity=64,
                                                    num_threads=4)   

n_epoch = 200

print_freq = 10
n_step_epoch = int(12916/batch_size)
print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
step=0
with tf.Graph().as_default():
    for epoch in range(n_epoch):
        start_time = time.time()
        for s in range(n_step_epoch):
            val, l1 = sess.run([x_train_batch, y1_train_batch])
            feed_dict = {x: val, y_: l1}
            feed_dict.update(network.all_drop)
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            train_loss, train_acc, n_batch = 0, 0, 0
            for s in range(n_step_epoch):
                val, l1 = sess.run([x_train_batch, y1_train_batch])
                dp_dict = tl.utils.dict_to_one(network.all_drop)
                feed_dict = {x: val, y_: l1}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                step+=1
                train_loss += err; train_acc += ac; n_batch += 1
            print("Epoch %d : Step %d-%d of %d took %fs" % (epoch+1, step, step + s, n_step_epoch, time.time() - start_time))
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))
            
            test_loss, test_acc, n_batch = 0, 0, 0
            for _ in range(int(3710/batch_size)):
                valt, l1t = sess.run([x_test_batch, y1_test_batch])
                dp_dict = tl.utils.dict_to_one(network.all_drop)
                feed_dict = {x: valt, y_: l1t}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc],feed_dict = feed_dict)
                test_loss += err; test_acc += ac; n_batch += 1
            print("   test loss: %f" % (test_loss/ n_batch))
            print("   test acc: %f" % (test_acc/ n_batch))
        #if (epoch + 1) % (print_freq) == 0:
            print("Save model " + "!"*10)
            tl.files.save_npz(network.all_params, name = "SingleType_V1_model_"+str(epoch+1)+".npz", sess=sess)
coord.request_stop()
coord.join(threads)
sess.close()


