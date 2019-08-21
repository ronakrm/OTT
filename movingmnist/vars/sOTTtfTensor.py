import numpy as np
import tensorflow as tf

from vars.tt_tf import tt_tf

class sOTTtfTensor(tt_tf):

    def __init__(self, shape, r, name='sOTT_Tens_default'):
        super(sOTTtfTensor,self).__init__(shape, r, name)
        init = tf.orthogonal_initializer()

        # setup variables
        self.Q = self.setupQ(init)
        self.U = self.setupU()

        # for debugging
        self.W = self.setupW()

    def setupQ(self, init):
        
        # only need R choose 2 parameters
        sparseshape = int(self.r*(self.r-1)/2)
        print(sparseshape)

        # get list of sparse indices for upper triangular minus diag
        # Get pairs of indices of positions
        indices = list(zip(*np.triu_indices(self.r,k=1)))
        indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

        Q = []
        self.vs = []
        for i in range(0, self.d):
            for j in range(0, self.n[i]):
                vname = self._name+str(i)+str(j)
                if i == 0 or i == self.d-1 or self.r == 1:
                    # Vector for first and last cores of TT
                    myvar = tf.get_variable(vname, shape=[self.r,1], initializer=init)
                    tmp = myvar
                else:
                    # sparse representation for skew symm matrix
                    myvar = tf.get_variable(vname, shape=[sparseshape,1], initializer=init)
                    
                    # dense rep
                    striu = tf.SparseTensor(indices=indices, values=tf.squeeze(myvar), dense_shape=[self.r, self.r])
                    triu = tf.sparse_add(striu, tf.zeros(striu.dense_shape)) 
                    
                    # skew symmetric
                    sksym = triu - tf.transpose(triu)

                    # Cayley transform to Orthogonal SO(r)
                    I = tf.eye(self.r)
                    tmp = tf.matmul(I - sksym , tf.matrix_inverse(I + sksym))

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
                tmp = tf.transpose(tmp, perm=[2,1,0])
            U.append( tmp )            
        return U