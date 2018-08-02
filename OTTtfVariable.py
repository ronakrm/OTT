import tensorflow as tf

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel, Product

class OTTtfVariable():

	def __init__(self, opt_n, out_n, r):
		self.out_n = out_n
		self.d = len(opt_n) # number of modes of tensor rep
		self.n = [0] + opt_n # dim of modes in tensor rep
		#self.r = [1] + opt_n[:-1] + [1] # TT ranks
		self.r = r
		print(self.r)
		print(self.n)
		print(self.d)
		self.setupQ()
		self.setupU()
		self.setupW()

		# self.setupManifold()
		# self.setupProblem()

	def setupQ(self):
		self.Q = []
		for i in range(0, self.d):
			self.Q.append( tf.Variable( tf.zeros([self.r[i]*self.n[i+1], self.r[i+1]]) ) )
		self.Q.append( tf.Variable(tf.zeros(self.r[self.d])) ) # R

	def setupU(self):
		self.U = []
		for i in range(0, self.d):
			self.U.append( tf.reshape(self.Q[i], [self.r[i], self.n[i+1], self.r[i+1]]) )
		self.U[self.d-1] = tf.einsum('abc,c->abc', self.U[self.d-1], self.Q[self.d])

	def setupW(self):
		self.W = self.U[0] # first
		for i in range(1, self.d): # second through last
			self.W = tf.tensordot(self.W, self.U[i], axes=1)
		self.W = tf.reshape(self.W, self.out_n)

	# def setupManifold(self):
	# 	PM = ()
	# 	for i in range(0, self.d):
	# 		PM = PM + (Stiefel(self.r[i]*self.n[i], self.r[i+1]),)
	# 	PM = PM + (Euclidean(self.r[d]),)
	# 	self.manifold = Product( PM )

	# def getProblem(self, cost):
	# 	problem = Problem(manifold=self.manifold, cost=cost, arg=self.getQ(), verbosity=2)
	# 	return problem

	def getQ(self):
		return self.Q

	def getW(self):
		return self.W

	def printDims(self):
		print('d: ', self.d)
		print('n: ', self.n)
		print('r: ', self.r)