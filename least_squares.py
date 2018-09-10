import tensorflow as tf
import numpy as np
import time

from OTTtfVariable import OTTtfVariable
from aOTTtfVariable import aOTTtfVariable
import utils as ut

from pymanopt import Problem
from pymanopt.solvers import StochasticGradient, TrustRegions
from pymanopt.manifolds import Product

if __name__ == "__main__":

    t0 = time.time()
    lr = 1e-4

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
    r = [1,5,5,5,1]
    #r = [1, max(n), max(n), max(n), 1]

    np.random.seed(0)

    N = 1
    X_data = np.random.uniform(size=[N,dx]).astype('float32')
    W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
    Y_data = np.matmul(X_data, W_gt) + 0.001*np.random.randn(N,dy)

    X = tf.placeholder(tf.float32, [None, dx])
    Y = tf.placeholder(tf.float32, [None, dy])

    W_hat = aOTTtfVariable(shape=[ny,nx], r=r)

    # Cost function is the sqaured test error
    Y_hat = tf.transpose(W_hat.mult(tf.transpose(X)))
    cost = tf.reduce_mean(tf.square(Y - Y_hat))

    # first-order, second-order
    #solver = TrustRegions(maxiter=10000)
    solver = StochasticGradient(stepsize=lr, logverbosity=2)

    # Product Manifold
    manifold = Product( W_hat.getManifoldList() )

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost, arg=W_hat.getQ(), verbosity=2)
    
    niters = 100
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    wopt = sess.run(W_hat.getQ())
    for i in range(0,niters):
        x_mb, y_mb = ut.next_batch(X_data, Y_data, 1)
        wopt, optlog = solver.solve(problem, x = wopt, feed_dict={X: x_mb, Y: y_mb})
        print(optlog['final_values']['f(x)'])
    t1 = time.time()
    print(t1 - t0)