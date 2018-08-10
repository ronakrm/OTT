import tensorflow as tf
import numpy as np

from OTTtfVariable import OTTtfVariable
from aOTTtfVariable import aOTTtfVariable
import utils as ut

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Product


from tensorflow.examples.tutorials.mnist import input_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":

#	mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

#	trainIms = mnist.train.images
#	trainLbs = mnist.train.labels

	dx = 784
	dy = 25
	dz = 10
	d1 = 6
	n1 = [4,7,4,7,5,5]
	r1 = [1,4,7,4,7,5,1]
	d2 = 4
	n2 = [5,5,2,5]
	r2 = [1,5,5,2,1]

	np.random.seed(0)

	N = 10
	X = np.random.randn(N,dx).astype('float32')
	W1_gt = np.random.randn(dx,dy).astype('float32')
	H = np.maximum(X.dot(W1_gt), 0) # + 0.01*np.random.randn(N,dy)
	W2_gt = np.random.randn(dy,dz).astype('float32')
	Z = sigmoid(H.dot(W2_gt))

	W1_hat = OTTtfVariable(opt_n=n1, out_dims=[dx,dy], r=r1)
	W2_hat = OTTtfVariable(opt_n=n2, out_dims=[dy,dz], r=r2)

	# Cost function is the sqaured test error
	H_hat =	tf.nn.relu(tf.matmul(X,W1_hat.getW()))
	Z_hat = tf.nn.sigmoid(tf.matmul(H_hat, W2_hat.getW()))
	cost = tf.reduce_mean(tf.square(Z - Z_hat))

	# first-order, second-order
	#solver = TrustRegions(maxiter=1000)
	solver = SteepestDescent(maxiter=1000)

	# Product Manifold

	manifold = Product( W1_hat.getManifoldList() + W2_hat.getManifoldList() )

	# Solve the problem with pymanopt
	args = W1_hat.getQ() + W2_hat.getQ()
	problem = Problem(manifold=manifold, cost=cost, arg=args, verbosity=2)
	wopt = solver.solve(problem)

	W1_est = ut.OTTreconst(wopt[:d1+1], d1, n1, r1, [dx,dy])
	W2_est = ut.OTTreconst(wopt[d1+1:], d2, n2, r2, [dy,dz])

	print('test')
	#tX = np.random.randn(1,dx).astype('float32')
	tX = X[0,]
	print( sigmoid( (np.maximum(tX.dot(W1_est),0)).dot(W2_est) ) )
	print(Z[0,])