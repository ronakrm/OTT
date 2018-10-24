import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class TTtfVariable(tt_tf):

    def __init__(self, shape, r, name='TT_Var_default'):
        super(TTtfVariable,self).__init__(shape, r, name)
        # core_stddev = self.calcCoreGlorotSTD(self.n_dim, self.d, self.r):
        # init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        init = tf.orthogonal_initializer()
        
        # setup variables
        self.Q = self.setupQ(init)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    def setupQ(self, init):
        Q = []
        for i in range(0, self.d):
            vname = self._name+str(i).zfill(4)
            myshape = [self.r_array[i], self.n_out[i], self.n_in[i], self.r_array[i+1]]
            tmp = tf.get_variable(name=vname, shape=myshape, initializer=init)
            Q.append(tmp)
        return Q


    def setupU(self):
        return self.Q