import numpy as np
import tensorflow as tf

a = tf.constant(3)
b = tf.constant(7)
c = tf.constant(11)
myvars = [a,b,c] # should be a list

i = tf.constant(0)

def cond(i, lvars):
	return tf.less(i, len(lvars))
def body(i, lvars):
	i = i + 1
	return tf.gather(lvars, i)

_, val = tf.while_loop(cond, body, [i, myvars])

sess = tf.Session()
print(sess.run(val))
