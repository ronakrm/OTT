#test edit
import tensorflow as tf
import numpy as np

from OTTtfVariable import OTTtfVariable

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

if __name__ == "__main__":

	# This is just a 5x3 matrix with full rank
	r = [1,3,1]
	n1 = 5
	n2 = 3
	n = [n1,n2]
	d = len(n)

	np.random.seed(0)

	# Generate random data
	X_gt = np.random.randn(n1,n2).astype('float32')

	X_hat = OTTtfVariable(opt_n=n, out_n=n, r=r)

	cost = tf.reduce_mean(tf.square(X_hat.getW() - X_gt)) # scalar

	# first-order, second-order
	#solver = TrustRegions()
	solver = SteepestDescent()

	# Product Manifold
	PM = ()
	for i in range(0,d):
		PM = PM + (Stiefel(r[i]*n[i], r[i+1]),)
	PM = PM + (Euclidean(r[d]),)
	manifold = Product( PM )

	# Solve the problem with pymanopt
	problem = Problem(manifold=manifold, cost=cost, arg=X_hat.getQ(), verbosity=2)
	wopt = solver.solve(problem)
	
	print(wopt)

	Uest = []
	for i in range(0,d):
		Uest.append( np.reshape(wopt[i],[r[i],n[i],r[i+1]]) )
	Uest.append(wopt[d])
	print('X_est:')
	print( np.einsum('abc,cde,e->bd', *Uest) )
	print('X_gt:')
	print(X_gt)
