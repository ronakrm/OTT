import tensorflow as tf
import numpy as np
import t3f

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

if __name__ == "__main__":

	# This is just a 5x3 matrix with full rank
	r = 3
	r0 = 1
	r1 = r
	r2 = 1
	tt_ranks = [r0,r1,r2]
	n1 = 5
	n2 = 3
	Ns = [n1,n2]
	d = len(Ns)


	# Generate random data
	X_gt = np.random.randn(n1,n2).astype('float32')
	X_gt3f = t3f.to_tt_tensor(X_gt, max_tt_rank=r)

	Q1 = tf.Variable(tf.zeros([r0*n1,r1])) # 5 X 3
	Q2 = tf.Variable(tf.zeros([r1*n2,r2])) # 9 X 1
	R2 = tf.Variable(tf.zeros([r2])) # 1 X 1

	U1 = tf.reshape(Q1,[r0,n1,r1]) # 1 X 5 X 3
	U2 = tf.reshape(Q2,[r1,n2,r2]) # 3 X 3 X 1
	U2 = tf.multiply(U2,R2) # 3 X 3 X 1

	# Cost function is the sqaured test error
	X_hat = tf.einsum('abc,cde->bd',U1,U2) # 5 X 3

	cost = tf.reduce_mean(tf.square(X_hat - t3f.full(X_gt3f))) # scalar

	# first-order, second-order
	#solver = TrustRegions()
	solver = SteepestDescent()

	# Product Manifold
	manifold = Product(( Stiefel(r0*n1, r1), Stiefel(r1*n2, r2), Euclidean(r2) ))
	#manifold = Product(( Euclidean(r0*n1, r1), Euclidean(r1*n2, r2), Euclidean(r2) ))

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=[Q1,Q2,R2], verbosity=2)
	wopt = solver.solve(problem)

	print('X_est:')
	print(np.einsum('abc,cde,e->bd', np.reshape(wopt[0],[r0,n1,r1]), np.reshape(wopt[1],[r1,n2,r2]), wopt[2]))
	print('X_gt:')
	print(X_gt)