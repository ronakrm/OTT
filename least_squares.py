import tensorflow as tf
import numpy as np

from OTTtfVariable import OTTtfVariable
from aOTTtfVariable import aOTTtfVariable
import utils as ut

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Product

import pprint

if __name__ == "__main__":


	# least squares
	# y = Wx + noise
	# y in R^625 (25*25)
	# x in R^784 (28*28)
	# W in 625x784
	dx = 784
	dy = 625
	nx = [4,7,4,7]
	ny = [5,5,5,5]
	n = map(lambda x,y:x*y, nx, ny)
	r = [1, max(n), max(n), max(n), 1]

	np.random.seed(0)

	N = 1
	X_data = np.random.uniform(size=[N,dx]).astype('float32')
	W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
	Y_data = np.matmul(X_data, W_gt)# + 0.01*np.random.randn(N,dy)

	W_hat = aOTTtfVariable(shape=[nx,ny], r=r)

	# Cost function is the sqaured test error
	Y_hat =	tf.matmul(X_data,W_hat.getW())
	cost = tf.reduce_mean(tf.square(Y_data - Y_hat))

	# first-order, second-order
	#solver = TrustRegions(maxiter=10000)
	solver = SteepestDescent(maxiter=10, logverbosity=2)

	# Product Manifold
	manifold = Product( W_hat.getManifoldList() )

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=W_hat.getQ(), verbosity=2)
	wopt, log = solver.solve(problem)
	# print('And here comes the log:\n\r')
	# pp = pprint.PrettyPrinter()
	# pp.pprint(log)
	# print('len of wopt:')
	# print(len(wopt))

	# W_est = ut.aOTTreconst(wopt, [nx, ny], r)

	# print( W_est )
	# print('W_gt:')
	# print(W_gt)

	# print('test')
	#tX = np.random.randn(1,dx).astype('float32')
	# tX = X_data[0,]
	# print(tX.dot(W_est))
	# print(tX.dot(W_gt))