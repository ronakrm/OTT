import numpy as np
import tensorflow as tf
import t3f
from TTtfVariable import TTtfVariable

ny = [1,4,3]
nx = [7,5,1]
r = [1,4,4,1]
#np.random_seed(0)
tf.set_random_seed(0)
B = np.random.uniform(size=[np.prod(nx),3]).astype('float32')
BB = tf.convert_to_tensor(B)

sess = tf.Session()
A = TTtfVariable(shape=[ny,nx], r=r, name='myvar')


tf.set_random_seed(0)
initializer = t3f.glorot_initializer([ny, nx], tt_rank=max(r))
AA = t3f.get_variable('AA', initializer=initializer) 
#AA = t3f.matrix_ones([ny,nx])
tf.set_random_seed(0)

sess.run(tf.global_variables_initializer())
C = sess.run(A.mult(BB))
CC = sess.run(t3f.matmul(AA,BB))

print(C)
print(CC)

print(np.linalg.norm(C))
print(np.linalg.norm(CC))