import tensorflow as tf
import numpy as np
import t3f

from TTtfVariable import TTtfVariable
import utils as ut

if __name__ == "__main__":


	# least squares
	# y = Wx + noise
	# y in R^25 (25*25)
	# x in R^784 (28*28)
	# W in 625x784
    dx = 784
    dy = 625
    nx = [4,7,4,7]
    ny = [5,5,5,5]
    n = map(lambda x,y:x*y, nx, ny)
    #r = [1, max(n), max(n), max(n), 1]
    r = [1, 25,32,25, 1]

    np.random.seed(0)
    tf.set_random_seed(0)

    N = 1
    X = np.random.randn(dx, 1).astype('float32')
    W_gt = np.random.uniform(size=[dy,dx]).astype('float32')
    Y = np.matmul(W_gt, X)# + 0.01*np.random.randn(N,dy)

    print(X.shape)

    W_myTT = TTtfVariable(shape=[ny,nx], r=r)
    initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
    #initializer = t3f.matrix_ones(shape=[ny,nx])
    W_t3f = t3f.get_variable('W_hat', initializer=initializer) 
	# Cost function is the sqaured test error
    XX = tf.convert_to_tensor(X)
    Y_myTT = W_myTT.mult(X)
    Y_t3f = t3f.matmul(W_t3f, XX)
    mycost = tf.reduce_mean(tf.square(Y - Y_myTT))
    t3fcost = tf.reduce_mean(tf.square(Y - Y_t3f))

    mysolver = tf.train.AdamOptimizer().minimize(mycost)
    t3fsolver = tf.train.AdamOptimizer().minimize(t3fcost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    for it in range(1000):
        _, myloss = sess.run([mysolver, mycost])
        _, t3floss = sess.run([t3fsolver, t3fcost])
        #if it % 100 == 0:
        print('Iter',it,'myloss',myloss,'t3floss',t3floss)
    print('Iter',it,'myloss',myloss,'t3floss',t3floss)