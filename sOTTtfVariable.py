import numpy as np
import tensorflow as tf

class sOTTtfVariable():

    def __init__(self, shape, r, name='sOTT_Var_default'):
        self.ns = np.array(shape)
        self.n_out = self.ns[0]
        self.n_in = self.ns[1]
        self.in_dim = np.prod(self.n_in)
        self.out_dim = np.prod(self.n_out)
        self.d = len(self.n_in) # number of modes of tensor rep
        self.r = r
        self._name = name

        # glorot init variable
        lamb = 2.0 / (self.in_dim + self.out_dim)
        stddev = np.sqrt(lamb)
        cr_exp = -1.0 / (2* self.d)
        var = np.prod(self.r ** cr_exp)
        core_stddev = stddev ** (1.0 / self.d) * var

        # setup variables
        # self.Q = self.setupQ(core_stddev)
        self.Q = self.setupQ(0.01)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    def setupQ(self, core_stddev):

        # only need R choose 2 parameters
        sparseshape = int(self.r*(self.r-1)/2)

        # get list of sparse indices for upper triangular minus diag
        # Get pairs of indices of positions
        indices = list(zip(*np.triu_indices(self.r,k=1)))
        indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

        Q = []
        self.vs = []
        # init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        init = tf.random_uniform_initializer(minval=0., maxval=1.)
        for i in range(0, self.d):
            for j in range(0, self.n_out[i]):
                for k in range(0, self.n_in[i]):
                    vname = self._name+str(i).zfill(4)+str(j).zfill(4)+str(k).zfill(4)
                    if i == 0 or i == self.d-1 or self.r == 1:
                        # Vector for first and last cores of TT
                        myvar = tf.get_variable(vname, shape=[self.r,1], initializer=init)
                        tmp = myvar
                    else:
                        # sparse representation for skew symm matrix
                        myvar = tf.get_variable(vname, shape=sparseshape, initializer=init)
                        # dense rep
                        tmp2 = tf.sparse_to_dense(sparse_indices=indices, output_shape=[self.r, self.r], \
                           sparse_values=myvar, default_value=0, \
                           validate_indices=True)
                        
                        # skew symmetric
                        tmp3 = tmp2 - tf.transpose(tmp2)

                        # Cayley transform to Orthogonal SO(r)
                        I = tf.eye(self.r)
                        tmp = tf.matmul(I - tmp3 , tf.matrix_inverse(I + tmp3))

                    Q.append( tmp )
                    self.vs.append(myvar)
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

    def setupW(self):
        W = self.U[0] # first
        # W = self.U[0]
        print(self.U[0].shape)
        for i in range(1, self.d): # second through last
            print(self.U[i].shape)
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
                new_data_shape = (-1, self.n_in[i-1], self.r)
                data = tf.reshape(data, new_data_shape)
        return tf.reshape(data, (self.out_dim, right_dim))     

    def getQ(self):
        return self.Q
    def getU(self):
        return self.U
    
    def getW(self):
        return self.W

    def getV(self):
        return self.vs