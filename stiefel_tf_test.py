import tensorflow as tf
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import utils as ut

from stiefel_ops import proj as stproj
from stiefel_ops import retract as stretr
# from stiefel_ops import aprx_retract as astretr

if __name__ == "__main__":

    m = 8 
    n = 8

    X = rnd.randn(m, n)
    X = la.qr(X)[0] # base point

    H = rnd.randn(m, n) # random ambient

    Hproj = H - X.dot(X.T.dot(H) + H.T.dot(X)) / 2 # gt projection

    sess = tf.Session()

    U = stproj(X, H)
    AA = sess.run(U) # my projection

    ut.assertClose(Hproj, AA, atol=1e-10, Opname='Projection')
    # print('Projection Passed.')
    
    U = U / tf.norm(U) # random tangent vector

    Ret = stretr(X,U)
    BB = sess.run(Ret) # my retraction

    myeye = BB.T.dot(BB)

    ut.assertClose(myeye, np.eye(n,n), atol=1e-10, Opname='Retraction')