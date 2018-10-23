import tensorflow as tf
import numpy as np
#import t3f
import matplotlib.pyplot as plt
import time

from TTtfVariable import TTtfVariable
import utils as ut

if __name__ == "__main__":

    t0 = time.time()
    N = 1
    batch_size = 1
    niters = 100000

    lr = 1e-3
    dx = 256
    dy = 256
    nx = [4,4,4,4]
    ny = [4,4,4,4]
    n = map(lambda x,y:x*y, nx, ny)
    r = [1,2,2,2,1]

    np.random.seed(13245)

    X_data = np.random.uniform(size=[N,dx]).astype('float32')
    W_gt = (2*np.random.uniform(size=[dx,dy]).astype('float32')-1)
    #W_gt = W_gt/np.linalg.norm(W_gt,2)
    Y_data = np.matmul(X_data, W_gt) #+ 0.001*np.random.randn(N,dy)

    X = tf.placeholder(tf.float32, [batch_size, dx])
    Y = tf.placeholder(tf.float32, [batch_size, dy])

    W_hat = TTtfVariable(shape=[ny,nx], r=r)

    Y_hat = tf.transpose(W_hat.mult(tf.transpose(X)))

    loss = tf.reduce_mean(tf.square(Y - Y_hat))

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    for i in range(0,niters):
        x_mb, y_mb = ut.next_batch(X_data, Y_data, batch_size)
        _, itloss = sess.run([opt, loss], feed_dict={X: x_mb, Y: y_mb})

        print(i,itloss)
    t1 = time.time()
    print('Took seconds:', t1 - t0)

    print(np.linalg.norm(sess.run(scale*W_hat.getW()),'fro'))
    print(np.linalg.norm(W_gt,'fro'))
    print(sess.run(scale*W_hat.getW()))
    print(W_gt)
