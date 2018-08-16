import numpy as np
import tensorflow as tf

class TTtfVariable():

	def __init__(self, shape, r):

		self.ns = np.array(shape)
		self.in_dim = np.prod(self.ns[0])
		self.out_dim = np.prod(self.ns[1])
		self.n = np.multiply(self.ns[0], self.ns[1])
		self.d = len(self.n) # number of modes of tensor rep
		self.r = r
		self.setupQ()
		self.setupW()

	def setupQ(self):
		self.Q = []
		for i in range(0, self.d):
			self.Q.append( tf.Variable( tf.random_uniform([self.r[i], self.n[i], self.r[i+1]]) ) )

	def setupW(self):
		self.W = self.Q[0] # first
		for i in range(1, self.d): # second through last
			self.W = tf.tensordot(self.W, self.Q[i], axes=1)
		self.W = tf.reshape(self.W, [self.in_dim, self.out_dim])

	def getQ(self):
		return self.Q

	def getW(self):
		return self.W

	def printDims(self):
		print('d: ', self.d)
		print('n: ', self.n)
		print('r: ', self.r)