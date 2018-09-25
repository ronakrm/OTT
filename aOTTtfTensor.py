import numpy as np
import tensorflow as tf

class aOTTtfTensor():

    def __init__(self, shape, r, name='aOTT_Tens_default'):
        self.n = shape
        self.n_dim = np.prod(self.n)
        self.d = len(self.n) # number of modes of tensor rep
        self.r = np.array(r)
        self._name = name

        # glorot init variable
        lamb = 2.0 / (self.n_dim)
        stddev = np.sqrt(lamb)
        cr_exp = -1.0 / (2* self.d)
        var = np.prod(self.r ** cr_exp)
        core_stddev = stddev ** (1.0 / self.d) * var

        # setup variables
        self.Q = self.setupQ(core_stddev)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    def setupQ(self, core_stddev):
        Q = []
        #init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        orth_init = tf.orthogonal_initializer()
        for i in range(0, self.d):
            for j in range(0, self.n[i]):
                vname = self._name+str(i)+str(j)
                if self.r[i+1] > self.r[i]:
                   myshape = [self.r[i+1], self.r[i]]
                else:
                   myshape = [self.r[i], self.r[i+1]]
                tmp = tf.get_variable(vname, shape=myshape, initializer=orth_init)
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

    def setupW(self):
        W = self.U[0] # first
        for i in range(1, self.d): # second through last
            W = tf.tensordot(W, self.U[i], axes=1)
        W = tf.reshape(W, self.n)
        return W   

    def getQ(self):
        return self.Q
    
    def getW(self):
        return self.W