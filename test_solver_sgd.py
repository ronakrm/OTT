import tensorflow as tf
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import StochasticGradient
from pymanopt.manifolds import Euclidean, Stiefel, Product


def next_batch(x, y, batch_size):
    '''
    Return a total of `batch_size` random samples and labels. 
    '''
    idx = np.arange(0 , x.shape[0])
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    xdata_shuffle = [x[ i] for i in idx]
    ydata_shuffle = [y[ i] for i in idx]

    return(np.asarray(xdata_shuffle),np.asarray(ydata_shuffle))

if __name__ == "__main__":

    lr = 1e-2

    dx = 5
    dy = 3
    N = 10

    np.random.seed(0)

    X_data = np.random.uniform(size=[N,dx]).astype('float32')
    W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
    Y_data = np.matmul(X_data, W_gt) + 0.001*np.random.randn(N,dy)

    X = tf.placeholder(tf.float32, [None, dx])
    Y = tf.placeholder(tf.float32, [None, dy])


    # Set up Estimate
    W1_hat = tf.Variable(tf.ones([dx,dx]))
    W2_hat = tf.Variable(tf.ones([dx,dy]))

    # Cost function is the sqaured test error
    W_hat = tf.matmul(W1_hat, W2_hat)
    Y_hat = tf.matmul(X, W_hat)
    cost = tf.reduce_mean(tf.square(Y - Y_hat))


    # Set up Estimate
    W1_hatA = tf.Variable(tf.ones([dx,dx]))
    W2_hatA = tf.Variable(tf.ones([dx,dy]))

    # Cost function is the sqaured test error
    W_hatA = tf.matmul(W1_hatA, W2_hatA)
    Y_hatA = tf.matmul(X, W_hatA)
    costA = tf.reduce_mean(tf.square(Y - Y_hatA))


    # first-order, second-order
    solver = StochasticGradient(stepsize=lr, logverbosity=2)
    
    # Product Manifold
    manifold = Product(( Euclidean(dx, dx), Euclidean(dx, dy) ))

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost, arg=[W1_hat, W2_hat], verbosity=2)

    # tf standard GD
    tf_gd_solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(costA)


    niters = 100
    wopt = [np.ones([dx,dx]), np.ones([dx,dy])] # initial point
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0,niters):
        x_mb, y_mb = next_batch(X_data, Y_data, 10)
        wopt, optlog = solver.solve(problem, x = wopt, feed_dict={X: x_mb, Y: y_mb})
        _ = sess.run(tf_gd_solver, feed_dict={X: x_mb, Y: y_mb})
        tmp = sess.run(costA, feed_dict={X: x_mb, Y: y_mb})
        print(optlog['final_values']['f(x)'], tmp)

    