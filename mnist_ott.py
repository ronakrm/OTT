import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from aOTTtfVariable import aOTTtfVariable
from stiefel_ops import proj, retract

from tensorflow.examples.tutorials.mnist import input_data

########## Parameters ##########

niters = 1000
batch_size = 32
lr = 1e-4
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

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

########## First Layer ##########

W1 = aOTTtfVariable(shape=[nh,nx], r=r, name='W1')
b1 = tf.get_variable('b1', shape=[625])

Xnorm =  tf.nn.l2_normalize(X, dim=1)
o = tf.nn.l2_normalize( W1.mult(tf.transpose(Xnorm)), dim=0 )

#o = W1.mult(tf.transpose(Xnorm))

f1 = tf.transpose(o) + b1
h1 = tf.nn.relu(f1)


########## Second Layer ##########

W2 = tf.get_variable('W2', shape=[625, 10])
b2 = tf.get_variable('b2', shape=[10])

h2 = tf.matmul(h1, W2) + b2


########## regular cross entropy loss, solver setup and run ##########

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h2))

opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

EucgradsNvars = opt.compute_gradients(loss, [b1, W2, b2])
myEucgrads = [(-1*lr*g, v) for g, v in EucgradsNvars]
Eucupdate = opt.apply_gradients(myEucgrads)

aOTTgradsNvars = opt.compute_gradients(loss, W1.getQ())
projd = [proj(var, grad) for grad, var in aOTTgradsNvars]
#projd = [proj(aOTTgradsNvars[k][1], aOTTgradsNvars[k][0]) for k in range(len(aOTTgradsNvars))]
retrd = [retract(aOTTgradsNvars[k][1], -1*lr*projd[k]) for k in range(len(aOTTgradsNvars))]

Steifupdate = [aOTTgradsNvars[k][1].assign(retrd[k]) for k in range(len(aOTTgradsNvars))]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print('Total number of parameters: ', nparams)

for it in range(niters):
    X_mb, y_mb = mnist.train.next_batch(batch_size)

    #_, itloss = sess.run([solver, loss], feed_dict={X: X_mb, Y: y_mb})
    _, _, itloss = sess.run([Eucupdate, Steifupdate, loss], feed_dict={X: X_mb, Y: y_mb})
    #$if it % 100 == 0:
    if 1==1:
        pred = sess.run(tf.argmax(tf.nn.sigmoid(h2), axis=1), feed_dict={X: X_mb})
        acc = np.argmax(y_mb, axis=1)
        batch_acc = sum(np.equal(pred,acc))/float(batch_size)
        print('Iter',it,'Loss',itloss, 'batch acc', batch_acc)
