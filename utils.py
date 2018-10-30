import numpy as np
import numpy.testing as np_testing

def assertClose(A, B, atol=1e-10, Opname=""):
    errstr = Opname + ' fails closeness with tolerance ' + str(atol) + '.'
    np_testing.assert_allclose(A, B, atol=atol, err_msg=errstr)
    passtr = Opname + ' passed closeness with tolerance ' + str(atol) + '.'
    print(passtr)
    return