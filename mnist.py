import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import os

from aOTTtfVariable import aOTTtfVariable
from qrOTTtfVariable import qrOTTtfVariable

from pymanopt import Problem
from pymanopt.solvers import StochasticGradient
from pymanopt.manifolds import Product, Euclidean

from tensorflow.examples.tutorials.mnist import input_data

########## Parameters ##########

niters = 100
batch_size = 32
lr = 1e-1
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

########## 0th Layer ##########
#X_norm = tf.layers.batch_normalization(X, training=True)

########## First Layer ##########

W1 = aOTTtfVariable(shape=[nh, nx], r=r1, name='W1')
b1 = tf.get_variable('b1', shape=[625])
#Wa = tf.get_variable('Wa', shape=[625,625])
#W11 = tf.matmul(W1.getW(), Wa)
#f1 = tf.matmul(X, W11) + b1

f1 = tf.transpose(W1.mult(tf.transpose(X))) + b1
#f1 = tf.transpose(W1.mult(tf.transpose(X_norm))) + b1
#f1_norm = tf.layers.batch_normalization(f1, training=True)
#h1 = tf.nn.relu(f1_norm)
h1 = tf.nn.relu(f1)

########## Second Layer ##########

W2 = tf.get_variable('W2', shape=[625, 10])
b2 = tf.get_variable('b2', shape=[10])
f2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.relu(f2)

########## Third Layer ##########

#W3 = tf.get_variable('W3', shape=[625, 10])
#b3 = tf.get_variable('b3', shape=[10])
#h3 = tf.matmul(h2, W3) + b3

########## regular cross entropy loss, solver setup and run ##########

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h2))

mList = W1.getManifoldList() + (Euclidean(625,10),) + (Euclidean(625),) + (Euclidean(10),)
manifold = Product( mList )

args = W1.getQ()
args.append(W2)
args.append(b1)
args.append(b2)
print(len(args))
print(len(list(mList)))
problem = Problem(manifold=manifold, cost=cost, arg=args, verbosity=2)

solver = StochasticGradient(stepsize=lr, logverbosity=2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print('Total number of parameters: ', nparams)

wopt = None
#with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
x_mb, y_mb = mnist.train.next_batch(batch_size)
for it in range(niters):
#    x_mb, y_mb = mnist.train.next_batch(batch_size)
#    xnormed = sess.run(X_norm, feed_dict={X: x_mb})
#    n, bins, patches = plt.hist(np.ndarray.tolist(xnormed), 50, normed=1, facecolor='green', alpha=0.75)
#    plt.show()
    wopt, optlog = solver.solve(problem, x = wopt, feed_dict={X: x_mb, Y: y_mb})
    print(wopt[0][0,0], np.linalg.norm(wopt[110]), np.linalg.norm(wopt[111]), np.linalg.norm(wopt[112]))
    #print(np.linalg.norm(sess.run(W2)))
    #print(np.linalg.norm(sess.run(b2)))
    # if it % 100 == 0:
    pred = sess.run(tf.argmax(tf.nn.sigmoid(h2), axis=1), feed_dict={X: x_mb})
    acc = np.argmax(y_mb, axis=1)
    batch_acc = sum(np.equal(pred,acc))/float(batch_size)
    #print('loss', optlog['final_values']['f(x)'], 'batch acc', batch_acc)       