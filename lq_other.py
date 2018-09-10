import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from OTTtfVariable import OTTtfVariable
from aOTTtfVariable import aOTTtfVariable
import utils as ut
from stiefel_ops import proj, retract

#from pymanopt import Problem
#from pymanopt.solvers import StochasticGradient, TrustRegions
#from pymanopt.manifolds import Product

if __name__ == "__main__":

    t0 = time.time()
    lr = 1e-3
    batch_size = 10
    niters = 10000

    # least squares
    # y = Wx + noise
    # y in R^625 (25*25)
    # x in R^784 (28*28)
    # W in 625x784
    dx = 784
    dy = 625
    nx = [4,7,4,7]
    ny = [5,5,5,5]
    n = map(lambda x,y:x*y, nx, ny)
    #r = [1, max(n), max(n), max(n), 1]
    r = [1,5,5,5,1]
    np.random.seed(0)

    N = 100
    X_data = 4*np.random.uniform(size=[N,dx]).astype('float32')
    W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
    #W_gt = W_gt/np.linalg.norm(W_gt)
    Y_data = 17*np.matmul(X_data, W_gt) + 0.001*np.random.randn(N,dy)

    X = tf.placeholder(tf.float32, [batch_size, dx])
    Y = tf.placeholder(tf.float32, [batch_size, dy])

    W_hat = aOTTtfVariable(shape=[ny,nx], r=r)
    #b = tf.get_variable('b1', shape=[625])

    #X_norm = tf.layers.batch_normalization(X,axis=1)
    #X_norm = tf.nn.l2_normalize(X) 
    # Cost function is the sqaured test error
    #Y_hat = tf.transpose(W_hat.mult(tf.transpose(X)))
    Y_hat = tf.transpose(W_hat.mult(tf.transpose(X)))
    scale = tf.get_variable('scale', shape=[1])
    #cost = tf.reduce_mean(tf.square(Y - Y_hat))
    cost = tf.reduce_mean(tf.square(Y - scale*Y_hat))

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr) # this lr is ignored

    aOTTgradsNvars = opt.compute_gradients(cost, W_hat.getQ())
    projd = [proj(var, grad) for grad, var in aOTTgradsNvars]
    retrd = [retract(aOTTgradsNvars[k][1], -1*lr*projd[k]) for k in range(len(aOTTgradsNvars))]

    Steifupdate = [aOTTgradsNvars[k][1].assign(retrd[k]) for k in range(len(aOTTgradsNvars))]

    EucgradsNvars = opt.compute_gradients(cost, [scale])
    myEucgrads = [(-1*lr*g, v) for g, v in EucgradsNvars]
    Eucupdate = opt.apply_gradients(myEucgrads)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0,niters):
        x_mb, y_mb = ut.next_batch(X_data, Y_data, batch_size)

        #_, itloss = sess.run([Steifupdate, cost], feed_dict={X: x_mb, Y: y_mb})
        _, _, itloss = sess.run([Steifupdate, Eucupdate, cost], feed_dict={X: x_mb, Y: y_mb})
        #n, bins, patches = plt.hist(np.ndarray.tolist(x_mb), 50, normed=1, facecolor='green', alpha=0.75)
        #plt.show()
        #wopt, optlog = solver.solve(problem, x = wopt, feed_dict={X: x_mb, Y: y_mb})
        #print(optlog['final_values']['f(x)'])
        #print(itloss)
        print(itloss,'b',sess.run(scale))
    t1 = time.time()
    print(t1 - t0)
