import numpy as np
import tensorflow as tf

from pymanopt import Problem
from pymanopt.manifolds import Stiefel

class aOTTtfVariable():

	def __init__(self, shape, r):
		self.ns = np.array(shape)
		self.in_dim = np.prod(self.ns[0])
		self.out_dim = np.prod(self.ns[1])
		self.n = np.multiply(self.ns[0], self.ns[1])
		self.d = len(self.n) # number of modes of tensor rep
		self.r = r
		self.setupQ()
		self.setupU()
		self.setupW()

		self.setupManifold()

	def setupQ(self):
		self.Q = []
		for i in range(0, self.d):
			for j in range(0, self.n[i]):
				if self.r[i+1] > self.r[i]:
					self.Q.append( tf.Variable( tf.random_uniform([self.r[i+1], self.r[i]]) ) )
				else:
					self.Q.append( tf.Variable( tf.random_uniform([self.r[i], self.r[i+1]]) ) )

	def setupU(self):
		self.U = []
		start = 0
		end = 0
		for i in range(0, self.d):
			end = end + self.n[i]
			tmp = tf.stack(self.Q[start:end], axis=1)
			start = end
			if self.r[i+1] > self.r[i]:
				self.U.append( tf.transpose(tmp, perm=[2, 1, 0]) )
			else:
				self.U.append( tmp )

	def setupW(self):
		self.W = self.U[0] # first
		for i in range(1, self.d): # second through last
			self.W = tf.tensordot(self.W, self.U[i], axes=1)
		self.W = tf.reshape(self.W, [self.in_dim, self.out_dim])

	def setupManifold(self):
	 	PM = ()
	 	for i in range(0, self.d):
	 		for j in range(0, self.n[i]):
	 			if self.r[i+1] > self.r[i]:
	 				PM = PM + (Stiefel(self.r[i+1], self.r[i]),)
	 			else:
	 				PM = PM + (Stiefel(self.r[i], self.r[i+1]),)
		self.manifoldList = PM

	# def getProblem(self, cost):
	# 	problem = Problem(manifold=self.manifold, cost=cost, arg=self.getQ(), verbosity=2)
	# 	return problem

	def getQ(self):
		return self.Q

	def getW(self):
		return self.W

	def getManifoldList(self):
		return self.manifoldList	

	def printDims(self):
		print('d: ', self.d)
		print('n: ', self.n)
		print('r: ', self.r)