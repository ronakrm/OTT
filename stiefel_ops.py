import numpy as np
import tensorflow as tf

def proj(X, U):
    # project U to Stiefel tangent space at X
    return U - multiprod(X, multisym(multiprod(multitransp(X), U)))

def retract(X, G):
    # retract tangent vector U on X to tangent space of Stiefel

    # first dim of X,G are number of stiefels (product space)
    #k = X.shape[0]
    #assert(k == G.shape[0])

    #if k == 1:
        # Calculate 'thin' qr decomposition of X + G
    Q, R = tf.qr(X + G)
    # Unflip any flipped signs
    XNew = tf.matmul(Q, tf.diag(tf.sign(tf.sign(tf.diag_part(R))+.5)))
    #else:
    #    XNew = X + G
    #    for i in xrange(k):
    #        q, r = tf.qr(XNew[i,:,:])
    #        XNew[i,:,:] = tf.dot(q, tf.diag(tf.sign(tf.sign(tf.diag(r))+.5)))
    return XNew


#### Following taken from pymanopt  #######

def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    #if len(tf.shape(A)) == 2:
    return tf.matmul(A, B)
        #return 'no not here'

    # Old (slower) implementation:
    # a = A.reshape(np.hstack([np.shape(A), [1]]))
    # b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    # return np.sum(a * b, axis=2)

    # Approx 5x faster, only supported by numpy version >= 1.6:
    #return tf.einsum('ijk,ikl->ijl', A, B)


def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    return tf.transpose(A)


def multisym(A):
    # Inspired by MATLAB multisym function by Nicholas Boumal.
    return 0.5 * (A + multitransp(A))