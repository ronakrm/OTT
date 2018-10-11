import numpy as np
import tensorflow as tf

class sOTTtfTensor():

    def __init__(self, shape, r, name='sOTT_Tens_default'):
        self.n = shape
        self.n_dim = np.prod(self.n)
        self.d = len(self.n) # number of modes of tensor rep
        self.r = r
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
        
        # only need R choose 2 parameters
        sparseshape = int(self.r*(self.r-1)/2)

        # get list of sparse indices for upper triangular minus diag
        # Get pairs of indices of positions
        indices = list(zip(*np.triu_indices(self.r,k=1)))
        indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

        Q = []
        self.vs = []
        init = tf.random_normal_initializer(mean=0., stddev=core_stddev, seed=0)
        for i in range(0, self.d):
            for j in range(0, self.n[i]):
                vname = self._name+str(i)+str(j)
                if i == 0 or i == self.d-1 or self.r == 1:
                    # Vector for first and last cores of TT
                    myvar = tf.get_variable(vname, shape=self.r, initializer=init)
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
            end = end + self.n[i]
            for j in range(0, self.n[i]):
                tmp.append(self.Q[start])
                start = start + 1
            start = end
            tmp = tf.stack(tmp, axis=1)
            if i==0:
                tmp = tf.expand_dims(tmp, 2)
                tmp = tf.transpose(tmp, perm=[2,1,0])
            elif i==self.d-1:
                tmp = tf.expand_dims(tmp, 2)
            elif self.r==1:
                tmp = tf.expand_dims(tmp, 2)
            U.append( tmp )            
            print(tmp.shape)
        return U

    def setupW(self):
        W =  self.U[0] # first
        for i in range(1, self.d): # second through last
            W = tf.tensordot(W, self.U[i], axes=1)
        return tf.reshape(W, self.n)

    def getQ(self):
        return self.Q
    
    def getW(self):
        return self.W

    def getV(self):
        return self.vs