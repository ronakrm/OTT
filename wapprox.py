import tensorflow as tf
import numpy as np
import time

from aOTTtfTensor import aOTTtfTensor
from stiefel_ops import gradStep


def ottTApprox(Wstar, niters, lr, myTTrank):
    ########## Parameters ##########
    n = np.array(Wstar.shape)
    r = (len(Wstar.shape)-1)*[myTTrank]
    r = [1] + r + [1]
    
    ########## First Layer ##########

    print(n)
    print(r)
    What = aOTTtfTensor(shape=np.array(Wstar.shape), r=r, name='W1')

    loss = tf.reduce_mean(tf.square(Wstar - What.getW()))

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

    AottGnVs = opt.compute_gradients(loss, What.getQ())
    g0 = AottGnVs[:][0]
    Steifupdate = [v.assign(gradStep(v, g, lr)) for g, v in AottGnVs]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    t0 = time.time()
    losses = []
    for it in range(niters):        
        _, itloss = sess.run([Steifupdate, loss])
        losses.append(itloss)
        
        print('Iter',it,'Loss',itloss)

    t1 = time.time()
    print('Took seconds:', t1 - t0)

    return t1, losses, itloss


if __name__ == "__main__":

    n = [5,3,6,8,16]

    Wstar = (2*np.random.uniform(size=n).astype('float32')-1)
    Wstar = Wstar/np.linalg.norm(Wstar)

    niters = 10000
    lr = 1#1e-1
    myTTranks = [1,5,10,20,50]
    tf.set_random_seed(0)

    losses = []
    for myTTrank in myTTranks:
        tf.reset_default_graph()
        mytime, _, loss = ottTApprox(Wstar, niters, lr, myTTrank)
        losses.append(loss)

    print(myTTranks)
    print(losses)