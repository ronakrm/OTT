import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class aOTTtfTensor(tt_tf):

    def __init__(self, shape, r, name='aOTT_Tens_default'):
        super(aOTTtfTensor,self).__init__(shape, r, name)
        init = tf.orthogonal_initializer()
        # core_stddev = self.calcCoreGlorotSTD(self.n_dim, self.d, self.r):
        # init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)

        # setup variables
        self.Q = self.setupQ(init)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    def setupQ(self, init):
        Q = []
        for i in range(0, self.d):
            for j in range(0, self.n[i]):
                vname = self._name+str(i)+str(j)
                if self.r[i+1] > self.r[i]:
                   myshape = [self.r[i+1], self.r[i]]
                else:
                   myshape = [self.r[i], self.r[i+1]]
                tmp = tf.get_variable(vname, shape=myshape, initializer=init)
                Q.append( tmp )
        return Q

    def setupU(self):
        U = []
        start = 0
        end = 0
        for i in range(0, self.d):
            tmp = []
            end = end + self.n[i]
            for j in range(0, self.n[i]):
                tmp.append(self.Q[start])
                start = start + 1
            start = end
            tmp = tf.stack(tmp, axis=1)
            if self.r[i+1] > self.r[i]:
                U.append( tf.transpose(tmp, perm=[2, 1, 0]) )
            else:
                U.append( tmp )            
        return U