import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

from aOTTtfVariable import aOTTtfVariable
from stiefel_ops import proj, retract

from tensorflow.examples.tutorials.mnist import input_data


def ottMNIST(niters, batch_size, lr, myTTrank):
    ########## Parameters ##########

    r = [1, myTTrank, myTTrank, myTTrank, 1]
    nx = [4, 7, 4, 7]
    nh = [5, 5, 5, 5]
    nz = [1,2,5,1]

    ########## Dataset ##########

    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    trainIms = mnist.train.images
    trainLbs = mnist.train.labels

    ########## Variables ##########

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    ########## First Layer ##########

    W1 = aOTTtfVariable(shape=[nh,nx], r=r, name='W1')
    b1 = tf.get_variable('b1', shape=[625])

    #Xnorm =  tf.nn.l2_normalize(X, dim=1)
    o = W1.mult(tf.transpose(X))

    #o = W1.mult(tf.transpose(Xnorm))

    f1 = tf.transpose(o) + b1
    h1 = tf.nn.relu(f1)

    #h2 = h1
    ########## Second Layer ##########

    W2 = tf.get_variable('W2', shape=[625, 10])
    b2 = tf.get_variable('b2', shape=[10])

    h2 = tf.matmul(h1, W2) + b2


    ########## regular cross entropy loss, solver setup and run ##########

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h2))

    predict = tf.argmax(tf.nn.sigmoid(h2), axis=1)
    gt = tf.argmax(Y, axis=1)
    correct = tf.equal(predict, gt)
    acc = tf.reduce_sum(tf.cast(correct, tf.float32))/float(batch_size)

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

    EucgradsNvars = opt.compute_gradients(loss, [b1, W2, b2])
    myEucgrads = [(-1*lr*g, v) for g, v in EucgradsNvars]
    Eucupdate = opt.apply_gradients(myEucgrads)

    aOTTgradsNvars = opt.compute_gradients(loss, W1.getQ())
    projd = [proj(var, grad) for grad, var in aOTTgradsNvars]
    #projd = [proj(aOTTgradsNvars[k][1], aOTTgradsNvars[k][0]) for k in range(len(aOTTgradsNvars))]
    retrd = [retract(aOTTgradsNvars[k][1], -1*lr*projd[k]) for k in range(len(aOTTgradsNvars))]

    Steifupdate = [aOTTgradsNvars[k][1].assign(retrd[k]) for k in range(len(aOTTgradsNvars))]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    X_test, y_test = mnist.test.next_batch(batch_size)
    t0 = time.time()
    losses = []
    batch_acc = []
    test_acc = []
    for it in range(niters):        
        X_mb, y_mb = mnist.train.next_batch(batch_size)

        _, itloss = sess.run([Steifupdate, loss], feed_dict={X: X_mb, Y: y_mb})
        _, itloss, b_acc = sess.run([Eucupdate, loss, acc], feed_dict={X: X_mb, Y: y_mb})
        losses.append(itloss)
        batch_acc.append(b_acc)
        
        #t_acc = sess.run(batch_acc, feed_dict={X: X_test, Y: y_test})
        #test_acc.append(t_acc)
        
        #if it % 100 == 0:
        print('Iter',it,'Loss',itloss, 'batch acc', b_acc)
            #print('Iter',it,'Loss',itloss, 'batch acc', b_acc, 'test acc', t_acc)

    t1 = time.time()
    print('Took seconds:', t1 - t0)

    return t1, losses, batch_acc, test_acc


if __name__ == "__main__":

    niters = 1000
    batch_size = 32
    lr = 1e-2
    myTTranks = [1,5,10,20,50]
    tf.set_random_seed(0)

    losses = []
    batchaccs = []
    for myTTrank in myTTranks:
        tf.reset_default_graph()
        mytime, loss, batch_acc, test_acc = ottMNIST(niters, batch_size, lr, myTTrank)
        losses.append(loss)
        batchaccs.append(batch_acc)

    ## plot data
    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1,niters+1,1), batchaccs[0], 'k-', label='rank=1')
    ax.plot(np.arange(1,niters+1,1), batchaccs[1], 'm-', label='rank=5')
    ax.plot(np.arange(1,niters+1,1), batchaccs[2], 'b-', label='rank=10')
    ax.plot(np.arange(1,niters+1,1), batchaccs[3], 'k:', label='rank=20')
    ax.plot(np.arange(1,niters+1,1), batchaccs[4], 'r-', label='rank=50')
    plt.legend()
    plt.show()