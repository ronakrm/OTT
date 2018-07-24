import tensorflow as tf
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

if __name__ == "__main__":


	h = 5
	w = 3
	r = w # full rank

	# Generate random data
	X_gt = np.random.randn(h,w).astype('float32')

	# Set up Estimate
	Q = tf.Variable(tf.zeros([h,h]))
	R = tf.Variable(tf.zeros([h,w]))
	X_hat = tf.matmul(Q,R)

	# Cost function is the sqaured test error
	cost = tf.reduce_mean(tf.square(X_hat - X_gt))

	# first-order, second-order
	solver = SteepestDescent()
	#solver = TrustRegions()
	
	# Product Manifold
	manifold = Product(( Stiefel(h, h), Euclidean(h, w) ))

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=[Q,R], verbosity=2)
	wopt = solver.solve(problem)

	print('X_est:')
	print(wopt[0].dot(wopt[1]))
	print('X_gt:')
	print(X_gt)