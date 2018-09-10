import tensorflow as tf
import numpy as np
import t3f

if __name__ == "__main__":


	# least squares
	# y = Wx + noise
	# y in R^625 (25*25)
	# x in R^784 (28*28)
	# W in 625x784
	dx = 784
	dy = 625
	nx = [4,7,4,7]
	ny = [5,5,5,5]
	n = map(lambda x,y:x*y, nx, ny)
	r = [1, max(n), max(n), max(n), 1]
	#r = [1, 5, 5, 5, 1]

	np.random.seed(0)

	N = 1
	X_data = np.random.uniform(size=[dx,N]).astype('float32')
	W_gt = np.random.uniform(size=[dy,dx]).astype('float32')
	Y_data = np.matmul(W_gt, X_data)# + 0.01*np.random.randn(N,dy)


	X = tf.placeholder(tf.float32, [dx, N])
	Y = tf.placeholder(tf.float32, [dy, N])

	initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
	W_hat = t3f.get_variable('W_hat', initializer=initializer) 

	# Cost function is the sqaured test error
	Y_hat =	t3f.matmul(W_hat, X)
	cost = tf.reduce_mean(tf.square(Y - Y_hat))

	solver = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cost)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
	print('Total number of parameters: ', nparams)

	for it in range(10000):
	    _, itloss = sess.run([solver, cost], feed_dict={X: X_data, Y: Y_data})
	    #if it % 100 == 0:
	    print('Iter',it,'Loss',itloss)