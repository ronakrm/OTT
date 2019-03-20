import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class TTtfVariable(tt_tf):

    def __init__(self, shape, r, name='TT_Var_default'):
        super(TTtfVariable,self).__init__(shape, r, name)
        # core_stddev = self.calcCoreGlorotSTD(self.n_dim, self.d, self.r):
        # init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        init = tf.orthogonal_initializer()
        # init = tf.glorot_uniform_initializer()
        
        # setup variables
        self.Q = self.setupQ(init)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    # def setupQ(self, init):
    #     Q = []
    #     for i in range(0, self.d):
    #         vname = self._name+str(i).zfill(4)
    #         myshape = [self.r_array[i], self.n_out[i], self.n_in[i], self.r_array[i+1]]
    #         tmp = tf.get_variable(name=vname, shape=myshape, initializer=init)
    #         Q.append(tmp)
    #     return Q


    # def setupU(self):
    #     return self.Q

    def setupQ(self, init):

        Q = []
        # self.vs = []
        for i in range(0, self.d):
            for j in range(0, self.n_out[i]):
                for k in range(0, self.n_in[i]):
                    vname = self._name+str(i).zfill(4)+str(j).zfill(4)+str(k).zfill(4)
                    myvar = None
                    if i == 0 or i == self.d-1 or self.r == 1:
                        # Vector for first and last cores of TT
                        myvar = tf.get_variable(vname, shape=[self.r,1], initializer=init)
                        # myvar = tf.nn.l2_normalize(myvar)
                        myvar = tf.nn.dropout(myvar, keep_prob=0.75)
                        tmp = myvar
                    else:
                        # sparse representation for skew symm matrix
                        myvar = tf.get_variable(vname, shape=[self.r,self.r], initializer=init)
                        myvar = tf.nn.dropout(myvar, keep_prob=0.75)
                        tmp = myvar
                    
                    #tmp = tmp/tf.linalg.norm(tmp, ord=2)
                    Q.append( tmp )
                    # self.vs.append(myvar)
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
            if i==0:
                tmp = tf.transpose(tmp, perm=[3,1,2,0])
            U.append( tmp )
        return U