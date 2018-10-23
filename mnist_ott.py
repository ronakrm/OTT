import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
np.set_printoptions(precision=4)

# import sys
# sys.path.append("/vars")
from vars.sOTTtfVariable import sOTTtfVariable
from utils import next_batch

from tensorflow.examples.tutorials.mnist import input_data


def projDet(v, g, lr):
    vnew = v - g*lr
    if vnew.shape[0] == vnew.shape[1]:
        if tf.linalg.det(vnew) < 0:
            vnew[:,0] = -1*vnew[:,0]
    return vnew

def ottMNIST(niters, batch_size, lr, myTTrank, trainX, trainY):
    ########## Parameters ##########

    # r = [1, myTTrank, myTTrank, myTTrank, 1]
    r = myTTrank
    nx = [4, 7, 4, 7]
    nh = [5, 5, 5, 5]
    nz = [1,2,5,1]

    ########## Variables ##########

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    ########## First Layer ##########

    # W1 = sOTTtfVariable(shape=[nh,nx], r=r, name='W1')
    W1 = sOTTtfVariable(shape=[nh,nx], r=r, name='W1')
    b1 = tf.get_variable('b1', shape=[625])

    c = tf.get_variable('c', shape=[1])
    o = tf.multiply(c, W1.mult(tf.transpose(X)))
    #o = W1.mult(tf.transpose(X))

    f1 = tf.transpose(o) + b1
    h1 = tf.nn.relu(f1)

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

    ########## Optimizer and Gradient Updates ##########

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    # opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    # Stiefel OTT Update
    # aOTTgradsNvars = opt.compute_gradients(loss, W1.getQ())
    # Steifupdate = [v.assign(projDet(v, g, lr=1)) for g, v in aOTTgradsNvars]

    # Euclidean Update
    # EucgradsNvars = opt.compute_gradients(loss, [b1, W2, b2])
    # EucgradsNvars = opt.compute_gradients(loss, [b1, W2, b2]+W1.getV())
    # myEucgrads = [(g, v) for g, v in EucgradsNvars]
    # Eucupdate = opt.apply_gradients(myEucgrads)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    t0 = time.time()
    losses = []
    batch_acc = []
    test_acc = []
    for it in range(niters):        
        X_mb, y_mb = next_batch(x=trainX, y=trainY, batch_size=batch_size)

        # _, itloss = sess.run([Steifupdate, loss], feed_dict={X: X_mb, Y: y_mb})
        # _, itloss, b_acc = sess.run([Eucupdate, loss, acc], feed_dict={X: X_mb, Y: y_mb})
        _, itloss, b_acc = sess.run([opt, loss, acc], feed_dict={X: X_mb, Y: y_mb})
        losses.append(itloss)
        batch_acc.append(b_acc)
        
        #t_acc = sess.run(batch_acc, feed_dict={X: X_test, Y: y_test})
        #test_acc.append(t_acc)
        
        #if it % 100 == 0:
        print('Iter',it,'Loss',itloss, '\tbatch acc', b_acc)
            #print('Iter',it,'Loss',itloss, 'batch acc', b_acc, 'test acc', t_acc)

    t1 = time.time()
    print('Took seconds:', t1 - t0)

    return t1, losses, batch_acc, test_acc, sess.run(W1.getQ())


if __name__ == "__main__":

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

    niters = 1000
    batch_size = 32
    lr = 1e-3
    # myTTranks = [1,2,5,10,20,50]
    r = 10
    # myTTranks = 6*[r]
    myTTranks = 100*[r]

    tf.set_random_seed(0)

    ########## Dataset ##########
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    trainX = mnist.train.images
    trainY = mnist.train.labels

    losses = []
    batchaccs = []
    for myTTrank in myTTranks:
        trainX = mnist.train.images
        trainY = mnist.train.labels
        tf.reset_default_graph()
        mytime, loss, batch_acc, test_acc, _ = ottMNIST(niters, batch_size, lr, myTTrank, trainX, trainY)
        losses.append(loss)
        batchaccs.append(batch_acc[niters-1])

    ## plot data
    # fig = plt.figure()
    # fig.show()
    # ax = fig.add_subplot(111)
    # ax.plot(np.arange(1,niters+1,1), batchaccs[0], 'k-', label='rank=1')
    # ax.plot(np.arange(1,niters+1,1), batchaccs[1], 'm-', label='rank=2')
    # ax.plot(np.arange(1,niters+1,1), batchaccs[2], 'g-', label='rank=5')
    # ax.plot(np.arange(1,niters+1,1), batchaccs[3], 'b-', label='rank=10')
    # ax.plot(np.arange(1,niters+1,1), batchaccs[4], 'c-', label='rank=20')
    # ax.plot(np.arange(1,niters+1,1), batchaccs[5], 'k:', label='rank=50')
    # plt.legend()
    # plt.show()

    print(batchaccs)
    n, bins, patches = plt.hist(batchaccs, bins=batch_size, facecolor='green', alpha=0.75)
    plt.xlabel('Batch Accuracy')
    plt.ylabel('Count')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$'))
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)

    plt.show()

    
