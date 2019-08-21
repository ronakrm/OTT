import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class OTTtfVariable(tt_tf):

    def __init__(self, shape, r, name='OTT_Var_default'):
        super(OTTtfVariable,self).__init__(shape, r, name)

        init = tf.orthogonal_initializer()

        self.Q = self.setupQ(init)
        self.R = tf.get_variable(name=self._name+str(self.d), \
                        initializer=tf.random_uniform([self.r_array[self.d],])) # R
        self.U = self.setupU()
        self.W = self.setupW()

    def setupQ(self, init):
        Q = []
        for i in range(0, self.d):
            vname = self._name+str(i)
            if self.r_array[i+1] > self.r_array[i]*self.n_out[i]*self.n_in[i]:
                myshape = [self.r_array[i+1], self.r_array[i]*self.n_out[i]*self.n_in[i]]
            else:
                myshape = [self.r_array[i]*self.n_out[i]*self.n_in[i], self.r_array[i+1]]
            tmp = tf.get_variable(name=vname, shape=myshape, initializer=init)
            Q.append( tmp )
        return Q

    def setupU(self):
        U = []
        for i in range(0, self.d):
            if self.r_array[i+1] > self.r_array[i]*self.n_out[i]*self.n_in[i]:
                U.append( tf.reshape(tf.transpose(self.Q[i]), [self.r_array[i], self.n_out[i], self.n_in[i], self.r_array[i+1] ]))
            else:
                U.append( tf.reshape(self.Q[i], [self.r_array[i], self.n_out[i], self.n_in[i], self.r_array[i+1]]) )
            U[-1] = tf.einsum('abcd,d->abcd', U[-1], self.R)
        return U

    def getR(self):
        return self.R