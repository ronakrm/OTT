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
    N = 1
    batch_size = 1
    niters = 100000

    # lr = 1e-4
    # dx = 625
    # dy = 784
    # nx = [5,5,5,5]
    # ny = [4,7,4,7]
    # n = map(lambda x,y:x*y, nx, ny)
    # #r = [1, max(n), max(n), max(n), 1]
    # r = [1,100,100,100,1]

    lr = 1e-2
    dx = 4
    dy = 4
    nx = [2,2]
    ny = [2,2]
    n = map(lambda x,y:x*y, nx, ny)
    #r = [1, max(n), max(n), max(n), 1]
    r = [1,10,1]
    

    np.random.seed(13245)

    #rank_normalizer = np.sqrt(4)*35+np.sqrt(3)*20

    X_data = np.random.uniform(size=[N,dx]).astype('float32')
    W_gt = 13*(2*np.random.uniform(size=[dx,dy]).astype('float32')-1)
    #W_gt = W_gt/np.linalg.norm(W_gt,2)
    Y_data = np.matmul(X_data, W_gt) #+ 0.001*np.random.randn(N,dy)

    #c = np.max(np.abs(W_gt))
    #c = 1

    X = tf.placeholder(tf.float32, [batch_size, dx])
    Y = tf.placeholder(tf.float32, [batch_size, dy])

    W_hat = aOTTtfVariable(shape=[ny,nx], r=r)
    #b = tf.get_variable('b1', shape=[625])

    #X_norm = tf.layers.batch_normalization(X,axis=1)
    #X_norm = tf.nn.l2_normalize(X) 
    # Cost function is the sqaured test error
    #Y_hat = tf.transpose(W_hat.mult(tf.transpose(X)))
    #I = tf.eye(dx)
    #W_full = W_hat.mult(I)
    #W_clip = tf.clip_by_norm(W_full, clip_norm=1.0)
    #Y_hat = tf.transpose(tf.matmul(W_clip, tf.transpose(X)))

    scale = tf.get_variable('scale', shape=[1], initializer=tf.ones_initializer())
    Y_hat = tf.abs(scale)*tf.transpose(W_hat.mult(tf.transpose(X)))
    # Y_hat = 13*tf.transpose(W_hat.mult(tf.transpose(X)))
    loss = tf.reduce_mean(tf.square(Y - Y_hat))
    #loss = tf.reduce_mean(tf.square(c*W_hat.getW() - tf.transpose(W_gt)))
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr) # this lr is ignored

    # Stiefel OTT Update
    aOTTgradsNvars = opt.compute_gradients(loss, W_hat.getQ())
    projd = [proj(var, grad) for grad, var in aOTTgradsNvars]
    retrd = [retract(aOTTgradsNvars[k][1], -1*lr*projd[k]) for k in range(len(aOTTgradsNvars))]
    Steifupdate = [aOTTgradsNvars[k][1].assign(retrd[k]) for k in range(len(aOTTgradsNvars))]

    # Euclidean Update
    EucgradsNvars = opt.compute_gradients(loss, [scale])
    myEucgrads = [(-1*lr*g, v) for g, v in EucgradsNvars]
    Eucupdate = opt.apply_gradients(myEucgrads)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    for i in range(0,niters):
        x_mb, y_mb = ut.next_batch(X_data, Y_data, batch_size)

        _, itloss = sess.run([Steifupdate, loss], feed_dict={X: x_mb, Y: y_mb})
        _, itloss = sess.run([Eucupdate, loss], feed_dict={X: x_mb, Y: y_mb})

        #for j in range(0,100):
#            _, itloss = sess.run([Steifupdate, loss], feed_dict={X: x_mb, Y: y_mb})
        #for j in range(0,100):
        #_, itloss = sess.run([Eucupdate, loss], feed_dict={X: x_mb, Y: y_mb})


        print(itloss,'\tscale\t',sess.run(scale))
    t1 = time.time()
    print('Took seconds:', t1 - t0)

    # sess.run([Steifupdate], feed_dict={X: x_mb, Y: y_mb})

    # vv = 0
    # for i in range(0,len(W_hat.getQ())):
    #     a = np.linalg.norm(sess.run(W_hat.getQ()[i]))
    #     if a > 1.1:
    #         print(W_hat.getQ()[i].name, a)
    #         print(W_hat.getQ()[i].shape)
    #         vv = vv + 1
    #     else:
    #         print('other shape: ', W_hat.getQ()[i].shape)
    print(np.linalg.norm(sess.run(scale*W_hat.getW()),'fro'))
    print(np.linalg.norm(W_gt,'fro'))
    print(sess.run(scale*W_hat.getW()))
    print(W_gt)
