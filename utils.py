import numpy as np
import numpy.testing as np_testing

from pymanopt import Problem
from pymanopt.manifolds import Product

def assertClose(A, B, atol=1e-10, Opname=""):
    errstr = Opname + ' fails closeness with tolerance ' + str(atol) + '.'
    np_testing.assert_allclose(A, B, atol=atol, err_msg=errstr)
    passtr = Opname + ' passed closeness with tolerance ' + str(atol) + '.'
    print(passtr)
    return

def makeProductManifold():
    return NotImplemented

def aOTTreconst(core_list, shape, r):
    ns = np.array(shape)
    in_dim = np.prod(ns[0])
    out_dim = np.prod(ns[1])
    n = np.multiply(ns[0], ns[1])
    d = len(n) # number of modes of tensor rep
    U = []
    start = 0
    end = 0
    for i in range(0, d):
        end = end + n[i]
        tmp = np.stack(core_list[start:end], axis=1)
        start = end
        if r[i+1] > r[i]:
            U.append( np.transpose(tmp, [2, 1, 0]) )
        else:
            U.append( tmp )

    W = U[0] # first
    for i in range(1, d): # second through last
        W = np.tensordot(W, U[i], axes=1)
    W = np.reshape(W, [in_dim, out_dim])

    return W

def OTTreconst(core_list, shape, r):
    ns = np.array(shape)
    in_dim = np.prod(ns[0])
    out_dim = np.prod(ns[1])
    n = np.multiply(ns[0], ns[1])
    d = len(n) # number of modes of tensor rep
    U = []
    for i in range(0, d):
        if r[i+1] > r[i]*n[i]:
            U.append( np.transpose(np.reshape(core_list[i], [r[i+1], n[i], r[i]]), [2,1,0] ) )
        else:
            U.append( np.reshape(core_list[i], [r[i], n[i], r[i+1]]) )
    U[d-1] = np.einsum('abc,c->abc', U[d-1], core_list[d])

    W = U[0] # first
    for i in range(1, d): # second through last
        W = np.tensordot(W, U[i], axes=1)
    W = np.reshape(W, [in_dim, out_dim])
    
    return W

def TTreconst(core_list, shape, r):
    ns = np.array(shape)
    in_dim = np.prod(ns[0])
    out_dim = np.prod(ns[1])
    n = np.multiply(ns[0], ns[1])
    d = len(n) # number of modes of tensor rep

    W = core_list[0] # first
    for i in range(1, d): # second through last
        W = np.tensordot(W, core_list[i], axes=1)
    W = np.reshape(W, [in_dim, out_dim])
    
    return W

def next_batch(x, y=None, batch_size=32):
    '''
    Return a total of `batch_size` random samples and labels. 
    '''
    idx = np.arange(0 , x.shape[0])
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    xdata_shuffle = [x[ i] for i in idx]

    if y is not None:
        ydata_shuffle = [y[ i] for i in idx]
        return(np.asarray(xdata_shuffle),np.asarray(ydata_shuffle))
    else:
        return(np.asarray(xdata_shuffle))