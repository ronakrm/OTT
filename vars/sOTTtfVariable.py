import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from vars.tt_tf import tt_tf

class sOTTtfVariable(tt_tf):

    def __init__(self, shape, r, name='sOTT_Var_default'):
        super(sOTTtfVariable,self).__init__(shape, r, name)
        init = tf.orthogonal_initializer()
        

        # setup variables
        self.Q = self.setupQ(init)
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
        # self.vs = []
        init = tf.orthogonal_initializer()
        for i in range(0, self.d):
            for j in range(0, self.n_out[i]):
                for k in range(0, self.n_in[i]):
                    vname = self._name+str(i).zfill(4)+str(j).zfill(4)+str(k).zfill(4)
                    myvar = None
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
                        
                        # tmp = tf.linalg.expm(sksym)

                        # Cayley transform to Orthogonal SO(r)
                        I = tf.eye(self.r)
                        tmp = tf.matmul(I - sksym , tf.matrix_inverse(I + sksym))

                    # Dropout
                    # tmp = tf.nn.dropout(tmp, keep_prob=0.8)
                    
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

    # def getV(self):
    #     return self.vs
