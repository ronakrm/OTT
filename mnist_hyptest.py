# mnist comparisons
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from mnist_ott import ottMNIST
from vae_ott import ottVAEMNIST

from aOTTtfVariable import aOTTtfVariable
from stiefel_ops import proj, retract

from tensorflow.examples.tutorials.mnist import input_data

def OTTstat(qlist1, qlist2):
    assert(len(qlist1)==len(qlist2))

    L = 0
    for i in range(0,len(qlist1)):
        L = L + np.trace( np.matmul(qlist1[i].T, qlist2[i] ) )
    return L


def Strain(trainX, trainY):
    niters = 1000
    batch_size = 32
    lr = 1e-3
    myTTrank = 10
    tf.set_random_seed(0)

    _, _, _, _, ottvars = ottVAEMNIST(niters, batch_size, lr, myTTrank, trainX, trainY)

    return ottvars

if __name__ == "__main__":

    mnist = input_data.read_data_sets('../../MNIST_data')
    trainX = mnist.train.images
    trainY = mnist.train.labels

    trainXAp = trainX[trainY==3,]
    trainYAp = np.ones(trainXAp.shape[0])
    trainXAn = trainX[trainY!=3,]
    trainYAn = np.zeros(trainXAn.shape[0])

    trainX = np.concatenate((trainXAp,trainXAn),axis=0)
    trainY = np.concatenate((trainYAp,trainYAn),axis=0)

    # trainXB = trainX[trainY==5,]
    # trainYB = trainY[trainY==5,]

    nperms = 10000
    alpha = 0.05

    #for permtest loop
    ls = []
    for n in range(0,nperms):

        # need a good way to get real null here

        # train network on grp 1, get vars
        qlist1 = Strain(trainXA, trainYA)
        tf.reset_default_graph()
        # train network on grp 2, get vars
        qlist2 = Strain(trainXB, trainYB)
        tf.reset_default_graph()

        L = OTTstat(qlist1, qlist2)
        ls.append(L)
        np.save('nulldist_3v3.npy',ls)
        
    # get threshold
    myhist = np.sort(ls)
    print(myhist)
    Lstar = myhist[int(np.floor(nperms*alpha))]

    # train network on grp 1, get vars
    qlist1 = Strain(trainXA, trainYA)
    tf.reset_default_graph()
    # train network on grp 2, get vars
    qlist2 = Strain(trainXB, trainYB)
    tf.reset_default_graph()

    Lhat = OTTstat(qlist1, qlist2)

    print(Lstar)
    print(Lhat)
    fig = plt.figure()
    fig.show()
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=myhist, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()