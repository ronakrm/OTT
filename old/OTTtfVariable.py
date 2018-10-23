import numpy as np
import tensorflow as tf

from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Stiefel

class OTTtfVariable():

    def __init__(self, shape, r, name='OTT_Var_default'):
        self.ns = np.array(shape)
        self.n_out = self.ns[0]
        self.n_in = self.ns[1]
        self.in_dim = np.prod(self.n_in)
        self.out_dim = np.prod(self.n_out)
        self.d = len(self.n_in) # number of modes of tensor rep
        self.r = np.array(r)
        self._name = name

        # glorot init variable
        lamb = 2.0 / (self.in_dim + self.out_dim)
        stddev = np.sqrt(lamb)
        cr_exp = -1.0 / (2* self.d)
        var = np.prod(self.r ** cr_exp)
        core_stddev = stddev ** (1.0 / self.d) * var

        self.Q = self.setupQ(core_stddev)
        # self.setupU()
        #self.setupW()

        self.setupManifold()

    def setupQ(self, core_stddev):
        Q = []
        #init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        orth_init = tf.orthogonal_initializer()
        for i in range(0, self.d):
            vname = self._name+str(i)
            if self.r[i+1] > self.r[i]*self.n_out[i]*self.n_in[i]:
                myshape = [self.r[i+1], self.r[i]*self.n_out[i]*self.n_in[i]]
            else:
                myshape = [self.r[i]*self.n_out[i]*self.n_in[i], self.r[i+1]]
            tmp = tf.get_variable(name=vname, shape=myshape, initializer=orth_init)
            Q.append( tmp )
        Q.append( tf.get_variable(name=self._name+str(self.d), \
                        initializer=tf.random_uniform([self.r[self.d],])) ) # R
        return Q

    # def setupU(self):
    #     self.U = []
    #     for i in range(0, self.d):
    #         if self.r[i+1] > self.r[i]*self.n[i]:
    #             self.U.append( tf.transpose(tf.reshape(self.Q[i], [self.r[i+1], self.n[i], self.r[i] ])) )
    #         else:
    #             self.U.append( tf.reshape(self.Q[i], [self.r[i], self.n[i], self.r[i+1]]) )
    #         self.U[self.d-1] = tf.einsum('abc,c->abc', self.U[self.d-1], self.Q[self.d])

    # def setupW(self):
    # 	self.W = self.U[0] # first
    # 	for i in range(1, self.d): # second through last
    # 		self.W = tf.tensordot(self.W, self.U[i], axes=1)
    # 	self.W = tf.reshape(self.W, [self.in_dim, self.out_dim])


    def mult(self, arg):
        right_dim = tf.shape(arg)[1]
        data  = tf.transpose(arg)
        data = tf.reshape(data, (-1, self.n_in[-1], 1))
        for i in reversed(range(self.d)):
            #cur = tf.get_variable(self.Q[i])
            cur = tf.reshape(self.Q[i], [self.r[i], self.n_out[i], self.n_in[i], self.r[i+1]])
            data = tf.einsum('aijb,rjb->ira', cur, data)
            if i > 0:
                new_data_shape = (-1, self.n_in[i-1], self.r[i])
                data = tf.reshape(data, new_data_shape)
        data = tf.multiply(data, self.Q[self.d])
        return tf.reshape(data, (self.out_dim, right_dim))        

    def setupManifold(self):
        PM = ()
        for i in range(0, self.d):
            if self.r[i+1] > self.r[i]*self.n_out[i]*self.n_in[i]:
                PM = PM + (Stiefel(self.r[i+1], self.r[i]*self.n_out[i]*self.n_in[i]),)
            else:
                PM = PM + (Stiefel(self.r[i]*self.n_out[i]*self.n_in[i], self.r[i+1]),)
        PM = PM + (Euclidean(self.r[self.d]),)
        self.manifoldList = PM

    def getManifoldList(self):
        return self.manifoldList

    def getQ(self):
        return self.Q