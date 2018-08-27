import numpy as np
import tensorflow as tf

class TTtfVariable():

    def __init__(self, shape, r, name='TT_Var_default'):
        self.ns = np.array(shape)
        self.no = self.ns[0]
        self.ni = self.ns[1]
        self.in_dim = np.prod(self.ni)
        self.out_dim = np.prod(self.no)
        #self.n = np.multiply(self.ns[0], self.ns[1])
        self.d = len(self.ni) # number of modes of tensor rep
        self.r = r
        self._name = name

        self.setupQ()
       # self.setupW()

    def setupQ(self):
        self.Q = []
        init = tf.glorot_uniform_initializer()
        for i in range(0, self.d):
            vname = self._name+str(i)
            self.Q.append( tf.get_variable(vname, \
                            shape=[self.r[i], self.no[i], self.ni[i], self.r[i+1]], \
                            initializer=init) )

    def setupW(self):
        self.W = self.Q[0] # first
        for i in range(1, self.d): # second through last
            self.W = tf.tensordot(self.W, self.Q[i], axes=1)
            print(self.W.get_shape())
#            self.W = tf.einsum('aijb,bklc->aijklc',self.W, self.Q[i])
        print('---------SHAPE-----------')
        print(self.W.get_shape())
        self.W = tf.reshape(self.W, [self.in_dim, self.out_dim])
        print(self.W.get_shape())

    def mult(self, arg):
        data = tf.transpose(arg)
        data = tf.reshape(data, (-1, self.ni[-1], 1))
        for i in reversed(range(self.d)):
            print(data.get_shape())
            cur = self.Q[i]
            data = tf.einsum('aijb,rjb->ira', cur, data)
            if i > 0:
                new_data_shape = (-1, self.ni[i-1], self.r[i])
                data = tf.reshape(data, new_data_shape)
        print(data.get_shape())
        return tf.reshape(data, [self.out_dim, arg.shape[1]])

    def getQ(self):
        return self.Q

    def getW(self):
        return self.W

    def printDims(self):
        print('d: ', self.d)
        print('n: ', self.n)
        print('r: ', self.r)