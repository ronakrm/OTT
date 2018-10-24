import numpy as np
import numpy.testing as np_testing
import matplotlib.pyplot as plt

# from pymanopt import Problem
# from pymanopt.manifolds import Product

def assertClose(A, B, atol=1e-10, Opname=""):
    errstr = Opname + ' fails closeness with tolerance ' + str(atol) + '.'
    np_testing.assert_allclose(A, B, atol=atol, err_msg=errstr)
    passtr = Opname + ' passed closeness with tolerance ' + str(atol) + '.'
    print(passtr)
    return


def rnn_plotter(seq1, frame_size, seq2=None):
    # takes in one or two [seqlen, vec_size] matrices and plots their 
    # image representations of size frame_size

    ncols = seq1.shape[0]
    if seq2 is None:
        nrows = 1
    else:
        nrows = 2

    for i in range(0,ncols):
        tmp = np.reshape(seq1[i,:], [frame_size, frame_size])
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(tmp)

    if seq2 is not None:
        for i in range(0,ncols):
            sbpstr = str(nrows)+str(ncols)+str(i+ncols+1)
            tmp = np.reshape(seq2[i,:], [frame_size, frame_size])
            plt.subplot(nrows, ncols, ncols+i+1)
            plt.imshow(tmp)

    plt.show()