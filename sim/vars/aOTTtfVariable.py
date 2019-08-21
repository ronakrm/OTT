import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class aOTTtfVariable(tt_tf):

    def __init__(self, shape, r, name='aOTT_Var_default'):
        super(aOTTtfVariable,self).__init__(shape, r, name)
        core_stddev = self.calcCoreGlorotSTD(self.in_dim*self.out_dim, self.d, self.r)
        init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        # init = tf.orthogonal_initializer()
        
        # setup variables
        self.Q = self.setupQ(init)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    def setupQ(self, init):
        Q = []
        for i in range(0, self.d):
            for j in range(0, self.n_out[i]):
                for k in range(0, self.n_in[i]):
                    vname = self._name+str(i).zfill(4)+str(j).zfill(4)+str(k).zfill(4)
                    if self.r_array[i+1] > self.r_array[i]:
                       myshape = [self.r_array[i+1], self.r_array[i]]
                    else:
                       myshape = [self.r_array[i], self.r_array[i+1]]
                    tmp = tf.get_variable(vname, shape=myshape, initializer=init)
                    Q.append( tmp )
        return Q

    def setupU(self):
        U = []
        start = 0
        end = 0
        for i in range(0, self.d):
            tmp = []
            for j in range(0, self.n_out[i]):
                end = end + self.n_in[i]
                tmp.append(tf.stack(self.Q[start:end], axis=1))
                start = end
            tmp = tf.stack(tmp, axis=1)
            if self.r_array[i+1] > self.r_array[i]:
                U.append( tf.transpose(tmp, perm=[3, 1, 2, 0]) )
            else:
                U.append( tmp )            
        return U