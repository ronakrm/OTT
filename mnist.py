import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import os

from aOTTtfVariable import aOTTtfVariable
from qrOTTtfVariable import qrOTTtfVariable

from pymanopt import Problem
from pymanopt.solvers import StochasticGradient
from pymanopt.manifolds import Product

from tensorflow.examples.tutorials.mnist import input_data

########## Parameters ##########

niters = 1000
batch_size = 32
lr = 1e-2
nx = [4, 7, 4, 7]
nh = [5, 5, 5, 5]
#nz = [1, 2, 5, 1]
n1 = map(lambda x,y:x*y, nx, nh)
#r1 = [1, max(n1), max(n1), max(n1), 1]
r1 = [1, 100, 100, 100, 1]
#n2 = map(lambda x,y:x*y, nh, nz)
#r2 = [1, max(n2), max(n2), max(n2), 1]

########## Dataset ##########

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

########## Variables ##########

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

########## First Layer ##########

W1 = qrOTTtfVariable(shape=[nx, nh], r=r1, name='W1')
b1 = tf.get_variable('b1', shape=[625])
#Wa = tf.get_variable('Wa', shape=[625,625])
#W11 = tf.matmul(W1.getW(), Wa)
#f1 = tf.matmul(X, W11) + b1

f1 = tf.matmul(X, W1.getW()) + b1
h1 = tf.nn.relu(f1)

########## Second Layer ##########

W2 = tf.get_variable('W2', shape=[625, 10])
b2 = tf.get_variable('b2', shape=[10])
h2 = tf.matmul(h1, W2) + b2

########## regular cross entropy loss, solver setup and run ##########

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h2))

manifold = Product( W1.getManifoldList() )

args = W1.getQ()
problem = Problem(manifold=manifold, cost=cost, arg=args, verbosity=2)

solver = StochasticGradient(stepsize=lr, logverbosity=2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print('Total number of parameters: ', nparams)

wopt = None
for it in range(niters):
    x_mb, y_mb = mnist.train.next_batch(batch_size)
    x_mb = x_mb + 1e-5
    wopt, optlog = solver.solve(problem, x = wopt, feed_dict={X: x_mb, Y: y_mb})

    # if it % 100 == 0:
    pred = sess.run(tf.argmax(tf.nn.sigmoid(h2), axis=1), feed_dict={X: x_mb})
    acc = np.argmax(y_mb, axis=1)
    batch_acc = sum(np.equal(pred,acc))/float(batch_size)
    print('loss', optlog['final_values']['f(x)'], 'batch acc', batch_acc)       