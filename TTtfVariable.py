import numpy as np
import tensorflow as tf

class TTtfVariable():

    def __init__(self, shape, r, name='TT_Var_default'):

        #self.varScope = tf.get_variable_scope()

        self.ns = np.array(shape)
        self.n_out = self.ns[0]
        self.n_in = self.ns[1]
        self.in_dim = np.prod(self.n_in)
        self.out_dim = np.prod(self.n_out)
        #self.n = np.multiply(self.ns[0], self.ns[1])
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

    def setupQ(self, core_stddev):
        Q = []
        init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        #init = tf.ones_initializer()
        for i in range(0, self.d):
            vname = self._name+str(i)
            myshape = [self.r[i], self.n_out[i], self.n_in[i], self.r[i+1]]
            tmp = tf.get_variable(name=vname, shape=myshape, initializer=init)
            Q.append(tmp)
        return Q

    def mult(self, arg):
        right_dim = tf.shape(arg)[1]
        data  = tf.transpose(arg)
        data = tf.reshape(data, (-1, self.n_in[-1], 1))
        for i in reversed(range(self.d)):
            #cur = tf.get_variable(self.Q[i])
            cur = self.Q[i]
            data = tf.einsum('aijb,rjb->ira', cur, data)
            if i > 0:
                new_data_shape = (-1, self.n_in[i-1], self.r[i])
                data = tf.reshape(data, new_data_shape)
        return tf.reshape(data, (self.out_dim, right_dim))

    def getQ(self):
        return self.Q

    def projQ(self):
        UU = []
        for i in range(0, self.d):
            U = self.Q[i]
            print(U.shape)
            tmpA = []
            for j in range(0, self.n_in[i]):
                tmp = []
                for k in range(0, self.n_out[i]):
                    if U[:,k,j,:].shape[0] < U[:,k,j,:].shape[1]:
                        q,_ = tf.qr(tf.transpose(U[:,k,j,:]))
                        tmp.append(tf.transpose(q))
                    else:
                        q,_ = tf.qr(U[:,k,j,:])
                        tmp.append(q)
                tmpA.append(tf.stack(tmp, axis=1))
            tmpA = tf.stack(tmpA, axis=2)
            UU.append(tmpA)
        return UU


                  
