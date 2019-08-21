import numpy as np
import tensorflow as tf

# Generic Tensorflow Tensor Train Super Class
class tt_tf():

    def __init__(self, shape, r, name='TT_TF_default'):

        self._name = name
        self.n = np.array(shape)
        
        # operator
        if len(self.n)==2:
            self.n_out = self.n[0]
            self.n_in = self.n[1]
            self.d = len(self.n_in)
            self.in_dim = np.prod(self.n_in)
            self.out_dim = np.prod(self.n_out)

        # basic tensor
        else:
            self.n_dim = np.prod(self.n)
            self.d = len(self.n)

        # fixed rank for all cores
        if np.array(r).size==1:
            self.r_array = np.array([1] + [r]*(self.d-1) + [1])
            self.r = r
        else:
            raise ValueError('Different ranks per core not supported.')
            # if np.array(r).size == self.d+1:
                # self.r = np.array(r)
            # else:
                # raise ValueError('Rank list provided does not match'
                                # 'tensor dimension for TT representation')

    def getQ(self):
        return self.Q
    
    def getU(self):
        return self.U

    def getW(self):
        return self.W

    def setupQ(self, initializer):
        return NotImplementedError()

    def setupU(self):
        return NotImplementedError()

    def setupG(self):
        G = []
        for i in range(self.d):
            G.append(tf.reshape(self.U[i],
                [self.n_out[i]*self.r_array[i+1], self.n_in[i]*self.r_array[i]]))
        return G

    def setupW(self):
        W = self.U[0] # first
        for i in range(1, self.d): # second through last
            W = tf.tensordot(W, self.U[i], axes=1)
        if len(self.n)==2:
            W = tf.reshape(W, [self.in_dim, self.out_dim])
        else:
            W = tf.reshape(W, self.n)
        # print(W)
        return W


    # for operator matrix multiplcation
    # old using einsum like T3F
    def mult(self, arg):
        right_dim = tf.shape(arg)[1]
        data  = tf.transpose(arg)
        data = tf.reshape(data, (-1, self.n_in[-1], 1))
        for i in reversed(range(self.d)):
            cur = self.U[i]
            data = tf.einsum('aijb,rjb->ira', cur, data)
            if i > 0:
                new_data_shape = (-1, self.n_in[i-1], self.r)
                data = tf.reshape(data, new_data_shape)
        return tf.reshape(data, (self.out_dim, right_dim))


    # for operator matrix multiplcation
    # def mult(self, arg):
    #     for i in range(self.d):            
    #         full = tf.reshape(full, [-1, self.r])
    #         core = tf.transpose(self.G[i], [1, 0])
    #         core = tf.reshape(core, [self.r, -1])
    #         full = tf.matmul(full, core)
    #     return full


    # for operator matrix multiplcation over timepoints (second tensor slice)
    def t_mult(self, arg, max_time):
        out = []
        for i in range(0, max_time):
            out.append(self.mult(tf.squeeze(arg[:,i,:])))
        return tf.stack(out, axis=1)


    # for glorot/xavier initialization
    def calcCoreGlorotSTD(self, n_dim, d, r):
        # glorot init variable
        lamb = 2.0 / (n_dim)
        stddev = np.sqrt(lamb)
        cr_exp = -1.0 / (2* d)
        var = np.prod(r ** cr_exp)
        core_stddev = stddev ** (1.0 / d) * var
        return core_stddev


    # def projQ(self):
    #     UU = []
    #     for i in range(0, self.d):
    #         U = self.Q[i]
    #         print(U.shape)
    #         tmpA = []
    #         for j in range(0, self.n_in[i]):
    #             tmp = []
    #             for k in range(0, self.n_out[i]):
    #                 if U[:,k,j,:].shape[0] < U[:,k,j,:].shape[1]:
    #                     q,_ = tf.qr(tf.transpose(U[:,k,j,:]))
    #                     tmp.append(tf.transpose(q))
    #                 else:
    #                     q,_ = tf.qr(U[:,k,j,:])
    #                     tmp.append(q)
    #             tmpA.append(tf.stack(tmp, axis=1))
    #         tmpA = tf.stack(tmpA, axis=2)
    #         UU.append(tmpA)
    #     return UU
