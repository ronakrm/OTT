## Compare all things
## least squares

import math
import numpy as np
import time

import tensorflow as tf
import t3f as t3f
from vars.TTtfVariable import TTtfVariable
from vars.aOTTtfVariable import aOTTtfVariable
import utils as ut
from stiefel_ops import gradStep

import matplotlib.pyplot as plt


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

def LQ(niters, batch_size, lr, rank, X_data, Y_data, model):
    tf.reset_default_graph()

    sess = tf.Session()
    losses = []
    if model=='riemt3f':
        X = tf.placeholder(tf.float32, [None, dx])
        Y = tf.placeholder(tf.float32, [None, dy])
        #### T3F RGD ####
        print('Starting T3F RGD...')
        # using riemannian projection implemented by t3f, compute a separate update
        # this requires projection/rounding which should be computationally intensive
        initializer = t3f.glorot_initializer([nx, ny], tt_rank=rank)
        W_t3f_rgd = t3f.get_variable('W_t3f_rgd', initializer=initializer)

        cost_t3f_rgd = tf.reduce_mean(0.5*tf.square(Y - t3f.matmul(X, W_t3f_rgd)))

        # least squares derivative
        grad = t3f.to_tt_matrix( tf.matmul(tf.transpose(Y - t3f.matmul(X, W_t3f_rgd)), -1*X ), shape=[nx, ny], max_tt_rank=rank )
        riemannian_grad = t3f.riemannian.project(grad, W_t3f_rgd)
        # norm_t3f_rgd = t3f.frobenius_norm(riemannian_grad, epsilon=1e-10)

        ### HARD CODED SLOWER RATE HERE BC OF DIVERGENCE
        train_step = t3f.assign(W_t3f_rgd, t3f.round(W_t3f_rgd - 0.1*lr * riemannian_grad, max_tt_rank=rank) )

        sess.run(tf.global_variables_initializer())
        nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Total number of parameters: ', nparams)

        t0 = time.time()
        # while(sess.run(tf.less(mingradnorm, norm_t3f_rgd), feed_dict={X: X_data, Y: Y_data})):
        i = 0
        while(i<=niters):
            i = i + 1
            x_mb, y_mb = next_batch(X_data, Y_data, batch_size)
            _, tmp = sess.run([train_step.op, cost_t3f_rgd], feed_dict={X: x_mb, Y: y_mb})
            losses.append(tmp)
            # print(sess.run(norm_t3f_rgd, feed_dict={X: X_data, Y: Y_data}))
            print(i,tmp)
            if tmp < mincost or np.isnan(tmp):
                break
        t1 = time.time()
        myT = t1 - t0

    elif model=='ott':
        X = tf.placeholder(tf.float32, [None, dx])
        Y = tf.placeholder(tf.float32, [None, dy])
        # #### Own EOTT GD ####
        print('Starting OTT...')
        W_EOTT_gd = aOTTtfVariable(shape=[ny,nx], r=rank)

        cost_eott_gd = tf.reduce_mean(0.5*tf.square(Y - tf.transpose(W_EOTT_gd.mult(tf.transpose(X)))))

        opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        # Manifold Update
        gW1 = opt.compute_gradients(cost_eott_gd, W_EOTT_gd.getQ())
        man_update = [v.assign(gradStep(X=v, G=g, lr=lr)) for g, v in gW1]


        t0 = time.time()
        sess.run(tf.global_variables_initializer())
        nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Total number of parameters: ', nparams)

        wopt = sess.run(W_EOTT_gd.getQ())
        i = 0
        while(i<=niters):
            i = i + 1
            x_mb, y_mb = next_batch(X_data, Y_data, batch_size)
            _, tmp = sess.run([man_update, cost_eott_gd], feed_dict={X: x_mb, Y: y_mb})
            # _, tmp = sess.run([Eucupdate, cost_eott_gd], feed_dict={X: x_mb, Y: y_mb})
            losses.append(tmp)
            print(i,tmp)
            if tmp < mincost or np.isnan(tmp):
                break
        t1 = time.time()
        myT = t1 - t0


    else:
        print('what model is that? unknown')
        return


    t1 = time.time()
    print('Took seconds:', myT)

    return myT, losses


if __name__ == "__main__":

    ######## PARAMS ########
    niters = 1000
    batch_size = 10
    lr = 1e-4
    myTTranks = [5,10,20]#,2,5,10,20,50]
    np.random.seed(28980)
    # mingradnorm = 1e-1
    mincost = 10

    # random init seed
    #np.random.seed(28980)
    np.random.seed(89431896)

    ########## Dataset ##########

    ######## DATA ########
    N = 10
    batch_size = 10
    dx = 784
    dy = 625
    nx = [4,7,4,7]
    ny = [5,5,5,5]

    n = map(lambda x,y:x*y, nx, ny)

    X_data = np.random.uniform(size=[N,dx]).astype('float32')
    W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
    Y_data = np.matmul(X_data, W_gt)# + 0.01*np.random.randn(N,dy)
    

    ott_losses = []
    ott_times = []
    for myTTrank in myTTranks:
        tf.reset_default_graph()
        mytime, loss = LQ(niters, batch_size, lr, myTTrank, X_data, Y_data, 'ott')
        ott_losses.append(loss)
        ott_times.append(mytime)

    riem_losses = []
    riem_times = []
    for myTTrank in myTTranks:
        tf.reset_default_graph()
        mytime, loss = LQ(niters, batch_size, lr, myTTrank, X_data, Y_data, 'riemt3f')
        riem_losses.append(loss)
        riem_times.append(mytime)


    np.savez('cr_ott_vs_riem_Rank.npz', ott_losses, riem_losses, ott_times, riem_times)

## plot data
pcolors = ['g','b','r','c','m','k']

fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
for i,t in enumerate(myTTranks):
    rlab = 'Riem, TT-r = '+str(t)
    olab = 'OTT, TT-r = '+str(t)
    rline = pcolors[i]+'--'
    oline = pcolors[i]+'-'
    ax.plot(range(0,len(riem_losses[i])), riem_losses[i], rline, label=rlab)
    ax.plot(range(0,len(ott_losses[i])), ott_losses[i], oline, label=olab)

ax.set_yscale('log')
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Least Squares MSE', fontsize=16)


plt.legend(loc='upper right')
plt.show()

print(ott_times)
print(riem_times)

# # Times
# print('total time in seconds')
# print(T_t3frgd)
# print(T_eott)
# # print(T_aott)

# print('num iters')
# print(len(t3frgd_loss))
# print(len(eott_loss))
# # print(len(aott_loss))

# print('time per iter')
# print(T_t3frgd/len(t3frgd_loss))
# print(T_eott/len(eott_loss))
# # print(T_aott/len(aott_loss))
