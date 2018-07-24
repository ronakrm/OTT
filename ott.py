import tensorflow as tf
import t3f
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

if __name__ == "__main__":

	r = 3
	r0 = 1
	r1 = r
	r2 = r
	r3 = r
	r4 = 1
	tt_ranks = [r0,r1,r2,r3,r4]
	n1 = r
	n2 = r
	n3 = r
	n4 = r
	Ns = [n1,n2,n3,n4]
	d = len(tt_ranks)-1


	# Generate random data
	X_gt = t3f.random_tensor(Ns, tt_rank=tt_ranks, mean=0., stddev=1.)

	Q1 = tf.Variable(tf.zeros([r0*n1,r1]))
	Q2 = tf.Variable(tf.zeros([r1*n2,r2]))
	Q3 = tf.Variable(tf.zeros([r2*n3,r3]))
	Q4 = tf.Variable(tf.zeros([r3*n4,r4]))
	R4 = tf.Variable(tf.zeros([r4]))

	U1 = tf.reshape(Q1,[r0,n1,r1])
	U2 = tf.reshape(Q2,[r1,n2,r2])
	U3 = tf.reshape(Q3,[r2,n3,r3])
	U4 = tf.reshape(Q4,[r3,n4,r4])
	U4 = tf.einsum('abc,c->abc',U4,R4)

	# Cost function is the sqaured test error
	X_hat = tf.einsum('abc,cde,efg,ghi->bdfh',U1,U2,U3,U4)

	cost = tf.reduce_mean(tf.square(X_hat - t3f.full(X_gt)))

	# first-order, second-order
	#solver = TrustRegions()
	solver = SteepestDescent()

	# Product Manifold
	manifold = Product(( Stiefel(r0*n1, r1), Stiefel(r1*n2, r2), Stiefel(r2*n3,r3), Stiefel(r3*n4,r4), Euclidean(r4) ))

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=[Q1,Q2,Q3,Q4,R4], verbosity=2)
	wopt = solver.solve(problem)