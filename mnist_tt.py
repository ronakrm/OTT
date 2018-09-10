import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import t3f
from TTtfVariable import TTtfVariable

from tensorflow.examples.tutorials.mnist import input_data

########## Parameters ##########

niters = 1000
batch_size = 32
lr = 0.01
myTTrank = 100
r = [1, myTTrank, myTTrank, myTTrank, 1]
nx = [4, 7, 4, 7]
nh = [5, 5, 5, 5]
tf.set_random_seed(0)

########## Dataset ##########

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

trainIms = mnist.train.images
trainLbs = mnist.train.labels

########## Variables ##########

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

########## First Layer ##########

b1 = tf.get_variable('b1', shape=[625])

# initialize the weight matrix rep'd as TT using t3f. equivalent to:
# W1 = tf.get_variable('W1', shape=[784,625]) 
#initializer = t3f.glorot_initializer([nx, nh], tt_rank=r)
#initializer = t3f.matrix_ones([nx, nh])
#W1 = t3f.get_variable('W1', initializer=initializer) 
#f1 = t3f.matmul(X, W1) + b1

W1 = TTtfVariable(shape=[nh,nx], r=r, name='W1')
f1 = tf.transpose(W1.mult(tf.transpose(X))) + b1

# do a ReLU approximation, equivalent to:
h1 = tf.nn.relu(f1)
#z1 = tf.distributions.Bernoulli(logits=10*f1,dtype=tf.float32).sample()
#h1 = tf.multiply(f1,z1)

########## Second Layer ##########

W2 = tf.get_variable('W2', shape=[625, 10])
b2 = tf.get_variable('b2', shape=[10])
h2 = tf.matmul(h1, W2) + b2


########## regular cross entropy loss, solver setup and run ##########

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h2))

solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

gradsNvars = opt.compute_gradients(loss, tf.trainable_variables())

update = opt.apply_gradients(gradsNvars)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print('Total number of parameters: ', nparams)

for it in range(niters):
    X_mb, y_mb = mnist.train.next_batch(batch_size)

    #_, itloss = sess.run([solver, loss], feed_dict={X: X_mb, Y: y_mb})
    _, itloss = sess.run([update, loss], feed_dict={X: X_mb, Y: y_mb})
    if it % 100 == 0:
        pred = sess.run(tf.argmax(tf.nn.sigmoid(h2), axis=1), feed_dict={X: X_mb})
        acc = np.argmax(y_mb, axis=1)
        batch_acc = sum(np.equal(pred,acc))/float(batch_size)
        print('Iter',it,'Loss',itloss, 'batch acc', batch_acc)