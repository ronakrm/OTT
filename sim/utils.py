import numpy as np
import numpy.testing as np_testing

def assertClose(A, B, atol=1e-10, Opname=""):
    errstr = Opname + ' fails closeness with tolerance ' + str(atol) + '.'
    np_testing.assert_allclose(A, B, atol=atol, err_msg=errstr)
    passtr = Opname + ' passed closeness with tolerance ' + str(atol) + '.'
    print(passtr)
    return

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