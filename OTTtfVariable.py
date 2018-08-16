import numpy as np
import tensorflow as tf

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from pymanopt.manifolds import Euclidean, Stiefel

class OTTtfVariable():

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
			if self.r[i+1] > self.r[i]*self.n[i]:
				self.Q.append( tf.Variable( tf.random_uniform([self.r[i+1], self.r[i]*self.n[i]]) ) )
			else:
				self.Q.append( tf.Variable( tf.random_uniform([self.r[i]*self.n[i], self.r[i+1]]) ) )
		self.Q.append( tf.Variable(tf.random_uniform([self.r[self.d],])) ) # R

	def setupU(self):
		self.U = []
		for i in range(0, self.d):
			if self.r[i+1] > self.r[i]*self.n[i]:
				self.U.append( tf.transpose(tf.reshape(self.Q[i], [self.r[i+1], self.n[i], self.r[i] ])) )
			else:
				self.U.append( tf.reshape(self.Q[i], [self.r[i], self.n[i], self.r[i+1]]) )
		self.U[self.d-1] = tf.einsum('abc,c->abc', self.U[self.d-1], self.Q[self.d])

	def setupW(self):
		self.W = self.U[0] # first
		for i in range(1, self.d): # second through last
			self.W = tf.tensordot(self.W, self.U[i], axes=1)
		self.W = tf.reshape(self.W, [self.in_dim, self.out_dim])

	def setupManifold(self):
		PM = ()
		for i in range(0, self.d):
			if self.r[i+1] > self.r[i]*self.n[i]:
				PM = PM + (Stiefel(self.r[i+1], self.r[i]*self.n[i]),)
			else:
				PM = PM + (Stiefel(self.r[i]*self.n[i], self.r[i+1]),)
		PM = PM + (Euclidean(self.r[self.d]),)
		self.manifoldList = PM

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