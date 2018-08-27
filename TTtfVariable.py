import numpy as np
import tensorflow as tf

class TTtfVariable():

    def __init__(self, shape, r, name='TT_Var_default'):

        #self.varScope = tf.get_variable_scope()

        self.ns = np.array(shape)
        self.no = self.ns[0]
        self.ni = self.ns[1]
        self.in_dim = np.prod(self.ni)
        self.out_dim = np.prod(self.no)
        #self.n = np.multiply(self.ns[0], self.ns[1])
        self.d = len(self.ni) # number of modes of tensor rep
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
       # self.setupW()

    def setupQ(self, core_stddev):
        #with tf.variable_scope("var_scope"):
        #with tf.variable_scope(self.varScope, reuse=True):
        Q = []
        #init = tf.glorot_uniform_initializer()
        init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        #init = tf.ones_initializer()
        for i in range(0, self.d):
            vname = self._name+str(i)
            tmp = tf.get_variable(name=vname, \
                            shape=[self.r[i], self.no[i], self.ni[i], self.r[i+1]], \
                            initializer=init)
            Q.append(tmp)
        return Q

#     def setupW(self):
#         self.W = self.Q[0] # first
#         for i in range(1, self.d): # second through last
#             self.W = tf.tensordot(self.W, self.Q[i], axes=1)
#             print(self.W.get_shape())
# #            self.W = tf.einsum('aijb,bklc->aijklc',self.W, self.Q[i])
#         print('---------SHAPE-----------')
#         print(self.W.get_shape())
#         self.W = tf.reshape(self.W, [self.in_dim, self.out_dim])
#         print(self.W.get_shape())

    def mult(self, arg):
        #with tf.variable_scope("var_scope", reuse=True):
        #with tf.variable_scope(self.varScope, reuse=True):
        right_dim = tf.shape(arg)[1]
        #if self.in_dim == arg.shape[1]:#tf.shape(arg)[1]:
            #right_dim = tf.shape(arg)[0]
        #    data = tf.transpose(arg)
        #elif self.in_dim == arg.shape[0]:#tf.shape(arg)[0]:
            #right_dim = tf.shape(arg)[1]
            #data = arg
        #else:
            #right_dim = -20
            #data = arg
            #rint('LJFSDKLFJSDLKFJ')
        data  = tf.transpose(arg)
        data = tf.reshape(data, (-1, self.ni[-1], 1))
        for i in reversed(range(self.d)):
            #cur = tf.get_variable(self.Q[i])
            cur = self.Q[i]
            data = tf.einsum('aijb,rjb->ira', cur, data)
            if i > 0:
                new_data_shape = (-1, self.ni[i-1], self.r[i])
                data = tf.reshape(data, new_data_shape)
        return tf.reshape(data, (self.out_dim, right_dim))

    def getQ(self):
        return self.Q

    def getW(self):
        return self.W

    def printDims(self):
        print('d: ', self.d)
        print('n: ', self.n)
        print('r: ', self.r)