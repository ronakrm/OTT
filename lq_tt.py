import tensorflow as tf
import numpy as np

from TTtfVariable import TTtfVariable
import utils as ut

if __name__ == "__main__":


	# least squares
	# y = Wx + noise
	# y in R^25 (25*25)
	# x in R^784 (28*28)
	# W in 625x784
	dx = 784
	dy = 625
	nx = [4,7,4,7]
	ny = [5,5,5,5]
	n = map(lambda x,y:x*y, nx, ny)
	r = [1, max(n), max(n), max(n), 1]
	#r = [1, 2,2,2, 1]

	np.random.seed(0)

	N = 1
	X = np.random.randn(N,dx).astype('float32')
	W_gt = np.random.uniform(size=[dx,dy]).astype('float32')
	Y = X.dot(W_gt)# + 0.01*np.random.randn(N,dy)

	W_hat = TTtfVariable(shape=[nx,ny], r=r)

	# Cost function is the sqaured test error
	Y_hat =	tf.matmul(X, W_hat.getW())
	cost = tf.reduce_mean(tf.square(Y - Y_hat))

	solver = tf.train.GradientDescentOptimizer(learning_rate=1e-12).minimize(cost)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
	print('Total number of parameters: ', nparams)

	for it in range(1000):
	    _, itloss = sess.run([solver, cost])
	    #if it % 100 == 0:
	    print('Iter',it,'Loss',itloss)