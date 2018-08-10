import tensorflow as tf
import numpy as np

from OTTtfVariable import OTTtfVariable
from aOTTtfVariable import aOTTtfVariable
import utils as ut

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Product

if __name__ == "__main__":


	# least squares
	# y = Wx + noise
	# y in R^25 (5*5)
	# x in R^784 (28*28)
	# W in 25x784
	dx = 784
	dy = 25
	d = 6
	n = [4,7,4,7,5,5]
	r = [1,4,7,4,7,5,1]

	np.random.seed(0)

	N = 10
	X = np.random.randn(N,dx).astype('float32')
	W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
	Y = X.dot(W_gt)# + 0.01*np.random.randn(N,dy)

	W_hat = OTTtfVariable(opt_n=n, out_dims=[dx,dy], r=r)

	# Cost function is the sqaured test error
	Y_hat =	tf.matmul(X,W_hat.getW())
	cost = tf.reduce_mean(tf.square(Y - Y_hat))

	# first-order, second-order
	#solver = TrustRegions(maxiter=10000)
	solver = SteepestDescent(maxiter=10000)

	# Product Manifold
	manifold = Product( W_hat.getManifoldList() )

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=W_hat.getQ(), verbosity=2)
	wopt = solver.solve(problem)

	W_est = ut.OTTreconst(wopt, d, n, r, [dx,dy])

	print( W_est )
	print('W_gt:')
	print(W_gt)

	print('test')
	#tX = np.random.randn(1,dx).astype('float32')
	tX = X[0,]
	print(tX.dot(W_est))
	print(tX.dot(W_gt))