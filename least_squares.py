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

	X = np.random.randn(1,dx).astype('float32')
	W_gt = np.random.randn(dx,dy).astype('float32')
	Y = X.dot(W_gt) #+ 0.01*np.random.randn(N,dy)

	W_hat = aOTTtfVariable(opt_n=n, out_dims=[dx,dy], r=r)

	# Cost function is the sqaured test error
	Y_hat =	tf.matmul(X,W_hat.getW())
	cost = tf.reduce_mean(tf.square(Y - Y_hat))

	# first-order, second-order
	#solver = TrustRegions()
	solver = SteepestDescent()

	# Product Manifold
	manifold = Product( W_hat.getManifoldList() )

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=W_hat.getQ(), verbosity=2)
	wopt = solver.solve(problem)

	W_est = ut.aOTTreconst(wopt, d, n, r, [dx,dy])

	# Uest = []
	# for i in range(0,d):
	# 	Uest.append( np.reshape(wopt[i],[r[i],n[i],r[i+1]]) )
	# Uest.append(wopt[d])
	# print('W_est:')
	# W_est = np.reshape( np.einsum('abc,cde,efg,ghi,ijk,klm,m->bdfhjl', *Uest), [dx,dy] )
	print( W_est )
	print('W_gt:')
	print(W_gt)

	print('test')
	print(X.dot(W_est))
	print(Y)