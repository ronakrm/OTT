import tensorflow as tf
import numpy as np
import time

from sOTTtfTensor import sOTTtfTensor


def sottTApprox(Wstar, n, niters, lr, myTTrank):

    print(n)
    print(myTTrank)
    What = sOTTtfTensor(shape=n, r=myTTrank)

    # loss = tf.reduce_mean(tf.square(Wstar.getW() - What.getW()))
    loss = tf.reduce_mean(tf.square(Wstar - What.getW()))

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    # EucgradsNvars = opt.compute_gradients(loss, What.getV())
    # myEucgrads = [(g, v) for g, v in EucgradsNvars]
    # Eucupdate = opt.apply_gradients(myEucgrads)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total number of parameters: ', nparams)

    t0 = time.time()
    losses = []
    for it in range(niters):        
        _, itloss = sess.run([opt, loss])
        losses.append(itloss)
        
        print('Iter',it,'Loss',itloss)

    t1 = time.time()
    print('Took seconds:', t1 - t0)

    return t1, losses, itloss


if __name__ == "__main__":

    n = [5,3,6,8,16]

    Wstar = (2*np.random.uniform(size=n).astype('float32')-1)
    Wstar = Wstar/np.linalg.norm(Wstar)    

    niters = 1000
    lr = 1e-1
    # myTTranks = [1,5,10,20,50]
    myTTranks = [5]
    tf.set_random_seed(0)

    losses = []
    for myTTrank in myTTranks:
        tf.reset_default_graph()
        # rtt = (len(n)-1)*[myTTrank]
        # rtt = [1] + rtt + [1]
        # Wstar = aOTTtfTensor(shape=n, r=rtt)
        mytime, _, loss = sottTApprox(Wstar, n, niters, lr, myTTrank)
        losses.append(loss)

    print(myTTranks)
    print(losses)