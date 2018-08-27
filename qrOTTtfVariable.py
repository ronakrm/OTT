import numpy as np
import tensorflow as tf

from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Euclidean

class qrOTTtfVariable():

    def __init__(self, shape, r, name='aOTT_Var_default'):
        self.ns = np.array(shape)
        self.in_dim = np.prod(self.ns[0])
        self.out_dim = np.prod(self.ns[1])
        self.n = np.multiply(self.ns[0], self.ns[1])
        self.d = len(self.n) # number of modes of tensor rep
        self.r = r
        self._name = name

        self.setupQ()
        self.setupU()
        self.setupW()

        self.setupManifold()

    def setupQ(self):
        orth_init = tf.orthogonal_initializer()
        unif_init = tf.glorot_uniform_initializer()
        self.Q = []
        for i in range(0, self.d):
            for j in range(0, self.n[i]):
                vname1 = self._name+str(i)+str(j)+'_Q'
                vname2 = self._name+str(i)+str(j)+'_R'
                if self.r[i+1] > self.r[i]:
                    tmp1 = tf.get_variable(vname1, \
                        shape=[self.r[i+1], self.r[i]], initializer=orth_init )
                    tmp2 = tf.get_variable(vname2, \
                        shape=[self.r[i], self.r[i]], initializer=unif_init)
                else:
                    tmp1 = tf.get_variable(vname1, \
                        shape=[self.r[i], self.r[i+1]], initializer=orth_init )
                    tmp2 = tf.get_variable(vname2, \
                        shape=[self.r[i+1], self.r[i+1]], initializer=unif_init)
                self.Q.append( tmp1 )
                self.Q.append( tmp2 )

    def setupU(self):

        self.A = []
        k = 0
        for i in range(0, self.d):
            for j in range(0, self.n[i]):
                self.A.append(tf.matmul(self.Q[2*k],self.Q[2*k+1]))
                k = k + 1

        self.U = []
        start = 0
        end = 0
        for i in range(0, self.d):
            end = end + self.n[i]
            tmp = tf.stack(self.A[start:end], axis=1)
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
                    PM = PM + (Euclidean(self.r[i], self.r[i]),)
                else:
                    PM = PM + (Stiefel(self.r[i], self.r[i+1]),)
                    PM = PM + (Euclidean(self.r[i+1], self.r[i+1]),)
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