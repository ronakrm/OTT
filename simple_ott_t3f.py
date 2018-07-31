import tensorflow as tf
import numpy as np
import t3f

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

if __name__ == "__main__":

	# This is just a 5x3 matrix with full rank
	r = [1,3,1]
	n1 = 5
	n2 = 3
	n = [0,n1,n2]
	d = len(n)-1

	np.random.seed(0)

	# Generate random data
	X_gt = np.random.randn(n1,n2).astype('float32')
	X_gt3f = t3f.to_tt_tensor(X_gt, max_tt_rank=3)

	# Set up orthogonal matrices
	Q = []
	for i in range(0,d):
		Q.append( tf.Variable( tf.zeros([r[i]*n[i+1],r[i+1]]) ) )
	Q.append( tf.Variable(tf.zeros(r[d])) ) # R

	# reshape for cost evaluation
	U = []
	for i in range(0,d):
		U.append( tf.reshape(Q[i],[r[i],n[i+1],r[i+1]]) )
	U[d-1] = tf.einsum('abc,c->abc',U[d-1],Q[d])

	# Cost function is the sqaured test error
	X_hat = tf.einsum('abc,cde->bd',*U) # 5 X 3

	cost = tf.reduce_mean(tf.square(X_hat - t3f.full(X_gt3f))) # scalar

	# first-order, second-order
	#solver = TrustRegions()
	solver = SteepestDescent()

	# Product Manifold
	PM = ()
	for i in range(0,d):
		PM = PM + (Stiefel(r[i]*n[i+1], r[i+1]),)
	PM = PM + (Euclidean(r[d]),)
	manifold = Product( PM )
	#manifold = Product(( Euclidean(r0*n1, r1), Euclidean(r1*n2, r2), Euclidean(r2) ))

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=Q, verbosity=2)
	wopt = solver.solve(problem)
	
	Uest = []
	for i in range(0,d):
		Uest.append( np.reshape(wopt[i],[r[i],n[i+1],r[i+1]]) )
	Uest.append(wopt[d])
	print(Uest)
	print(len(Uest))
	print('X_est:')
	print( np.einsum('abc,cde,e->bd', *Uest) )
	print('X_gt:')
	print(tf.Session().run(t3f.full(X_gt3f)))