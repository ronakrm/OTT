import numpy as np
import tensorflow as tf
import time
from tensorflow.python.framework import ops

if __name__ == "__main__":

    d = 5#9#2#4#8#7
    lr = 1e-1
    niters = 10000

    ### GROUND TRUTH DATA
    Xrand = np.random.uniform(size=[d,d]).astype('float32')
    Qt,_ = np.linalg.qr(Xrand)
    if np.linalg.det(Qt) < 0:
        print(Qt)
        Qt[:,0] = -1*Qt[:,0]
    print(np.linalg.det(Qt))
    Q = tf.placeholder(tf.float32, [d,d])

    sparseshape = int(d*(d-1)/2)

    # get list of sparse indices for upper triangular minus diag
    # Get pairs of indices of positions
    indices = list(zip(*np.triu_indices(d ,k=1)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    init = tf.random_uniform_initializer(minval=0., maxval=1.)
    # sparse representation for skew symm matrix
    myvar = tf.get_variable('myvar', shape=sparseshape, initializer=init)

    # dense rep
    # triu = tf.sparse_to_dense(sparse_indices=indices, output_shape=[d,d], \
       # sparse_values=myvar, default_value=0, \
       # validate_indices=True)
    # triu = tf.scatter_nd(indices=indices, updates=myvar, shape=tf.constant([d,d]))

    # to get around sparse gradient issue
    striu = tf.SparseTensor(indices=indices, values=myvar, dense_shape=[d, d])
    triu = tf.sparse_add(striu, tf.zeros(striu.dense_shape)) 
    
    # skew symmetric
    A = triu - tf.transpose(triu)

    # Cayley transform to Orthogonal SO(r)
    # I = tf.eye(d)
    # Qhat = tf.matmul(I - A , tf.matrix_inverse(I + A))

    ## OR matrix exponential ##
    @ops.RegisterGradient("MatrixExponential")
    def _expm_grad(op, grad):
    # We want the backward-mode gradient (left multiplication).
    # Let X be the NxN input matrix.
    # Let J(X) be the the N^2xN^2 complete Jacobian matrix of expm at X.
    # Let Y be the NxN previous gradient in the backward AD (left multiplication)
    # We have
    # unvec( ( vec(Y)^T . J(X) )^T )
    #   = unvec( J(X)^T . vec(Y) )
    #   = unvec( J(X^T) . vec(Y) )
    # where the last part (if I am not mistaken) holds in the case of the
    # exponential and other matrix power series.
    # It can be seen that this is now the forward-mode derivative
    # (right multiplication) applied to the Jacobian of the transpose.
            grad_func = lambda x, y: scipy.linalg.expm_frechet(x, y, compute_expm=False)
            return tf.py_func(grad_func, [tf.transpose(op.inputs[0]), grad], tf.float64) # List of one Tensor, since we have one input
    Qhat = tf.linalg.expm(A)

    #loss = -1*tf.trace(tf.matmul(tf.transpose(Q), Qhat))#(Q - Qhat))
    loss = tf.reduce_mean(tf.square(Q - Qhat))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)    
    
    t0 = time.time()
    for i in range(0,niters):
        _, itloss = sess.run([opt, loss], feed_dict={Q: Qt})
        print(i,itloss)
    t1 = time.time()
    print('Took seconds:', t1 - t0)
    print('Qhat',sess.run(Qhat))
    print('Qt',Qt)
    print('Qt*Qt^T',np.matmul(Qt,Qt.T))
    print('Qhat*Qhat^T',sess.run(tf.matmul(Qhat,tf.transpose(Qhat))))
    print('Qt*Qhat^T',sess.run(tf.matmul(Q,tf.transpose(Qhat)), feed_dict={Q:Qt}))
    print('norm Qt*Qhat^T',sess.run(tf.linalg.norm(tf.matmul(tf.transpose(Qhat),Q)), feed_dict={Q:Qt}))
