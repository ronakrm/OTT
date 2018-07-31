import tensorflow as tf
import numpy as np
import t3f

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

if __name__ == "__main__":


	# least squares
	# y = Wx + noise
	# y in R^25 (5*5)
	# x in R^784 (28*28)
	# W in 25x784
	N = 10
	dx = 4
	dy = 4
	d = 4
	n = [0,2,2,2,2]
	r = [1,2,2,2,1]

	np.random.seed(0)

	X = np.random.randn(N,dx).astype('float32')
	W_gt = np.random.randn(dx,dy).astype('float32')
	Y = X.dot(W_gt) #+ 0.01*np.random.randn(N,dy)


	# Generate random init
	#initializer = t3f.glorot_initializer([[5, 5, 5, 5], [4, 7, 4, 7]], tt_rank=7)
	#W = t3f.get_variable('W', initializer=initializer) 

	Q = []
	for i in range(0,d):
		Q.append( tf.Variable( tf.zeros([r[i]*n[i+1],r[i+1]]) ) )
	Q.append( tf.Variable(tf.zeros(r[d])) ) # R

	#Q1 = tf.Variable(tf.zeros([r0*dx,r1])) # 5 X 3
	#Q2 = tf.Variable(tf.zeros([r1*dy,r2])) # 9 X 1
	#R2 = tf.Variable(tf.zeros([r2])) # 1

	U = []
	for i in range(0,d):
		U.append( tf.reshape(Q[i],[r[i],n[i+1],r[i+1]]) )
	U[d-1] = tf.einsum('abc,c->abc',U[d-1],Q[d])

	#U1 = tf.reshape(Q1,[r0,dx,r1]) # 1 X 5 X 3
	#U2 = tf.reshape(Q2,[r1,dy,r2]) # 3 X 3 X 1
	#U2 = tf.einsum('abc,c->abc',U2,R2) # 3 X 3 X 1

	W_hTT = tf.einsum('abc,cde,efg,ghi->bdfh',*U)
	W_hat = tf.reshape(W_hTT, [dx, dy])

	# Cost function is the sqaured test error
	#W_hat = tf.einsum('abc,cde->bd',U1,U2) # dx x dy
	Y_hat =	tf.matmul(X,W_hat)
	cost = tf.reduce_mean(tf.square(Y - Y_hat))

	# first-order, second-order
	#solver = TrustRegions()
	solver = SteepestDescent()

	# Product Manifold
	PM = ()
	for i in range(0,d):
		PM = PM + (Stiefel(r[i]*n[i+1], r[i+1]),)
	PM = PM + (Euclidean(r[d]),)
	manifold = Product( PM )
	#manifold = Product(( Stiefel(r0*dx, r1), Stiefel(r1*dy, r2), Euclidean(r2) ))
	#manifold = Product(( Euclidean(r0*n1, r1), Euclidean(r1*n2, r2), Euclidean(r2) ))

	# Solve the problem with pymanopt
	#args = Q.append(R)
	problem = Problem(manifold=manifold, cost=cost, arg=Q, verbosity=2)
	wopt = solver.solve(problem)

	Uest = []
	for i in range(0,d):
		Uest.append( np.reshape(wopt[i],[r[i],n[i+1],r[i+1]]) )
	Uest.append(wopt[d])
	print('W_est:')
	W_est = np.reshape( np.einsum('abc,cde,efg,ghi,i->bdfh', *Uest), [dx,dy] )
	print( W_est )
	print('W_gt:')
	print(W_gt)

	print('test')
	tX = np.random.randn(3,dx).astype('float32')
	print(tX.dot(W_est))
	print(tX.dot(W_gt))