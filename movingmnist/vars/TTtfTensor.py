import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class TTtfTensor(tt_tf):

    def __init__(self, shape, r, name='TT_Var_default'):
        super(TTtfTensor,self).__init__(shape, r, name)
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
            vname = self._name+str(i).zfill(4)
            myshape = [self.r[i], self.n[i], self.r[i+1]]
            tmp = tf.get_variable(name=vname, shape=myshape, initializer=init)
            Q.append(tmp)
        return Q

    def setupU(self):
        return self.Q