import numpy as np
import tensorflow as tf

class aOTTtfVariable():

    def __init__(self, shape, r, name='aOTT_Var_default'):
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
            for j in range(0, self.n_out[i]):
                for k in range(0, self.n_in[i]):
                    vname = self._name+str(i).zfill(4)+str(j).zfill(4)+str(k).zfill(4)
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
            for j in range(0, self.n_out[i]):
                end = end + self.n_in[i]
                tmp.append(tf.stack(self.Q[start:end], axis=1))
                start = end
            tmp = tf.stack(tmp, axis=1)
            if self.r[i+1] > self.r[i]:
                U.append( tf.transpose(tmp, perm=[3, 1, 2, 0]) )
            else:
                U.append( tmp )            
        return U

    def setupW(self):
        W = self.U[0] # first
        for i in range(1, self.d): # second through last
            W = tf.tensordot(W, self.U[i], axes=1)
        W = tf.reshape(W, [self.in_dim, self.out_dim])
        return W

    def mult(self, arg):
        right_dim = tf.shape(arg)[1]
        data  = tf.transpose(arg)
        data = tf.reshape(data, (-1, self.n_in[-1], 1))
        for i in reversed(range(self.d)):
            cur = self.U[i]
            data = tf.einsum('aijb,rjb->ira', cur, data)
            if i > 0:
                new_data_shape = (-1, self.n_in[i-1], self.r[i])
                data = tf.reshape(data, new_data_shape)
        return tf.reshape(data, (self.out_dim, right_dim))     

    def getQ(self):
        return self.Q
    def getU(self):
        return self.U
    
    def getW(self):
        return self.W