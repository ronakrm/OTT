## Compare all things
## least squares

import math
import numpy as np
import tensorflow as tf
import t3f as t3f
from TTtfVariable import TTtfVariable
from OTTtfVariable import OTTtfVariable
from aOTTtfVariable import aOTTtfVariable
import utils as ut

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pymanopt.manifolds import Product

import matplotlib.pyplot as plt

######## PARAMS ########
np.random.seed(28980)
# mingradnorm = 1e-1
mincost = 5
max_iters = 1000
sess = tf.Session()

pymanopt_solver = SteepestDescent(maxiter=max_iters, mincost=mincost, logverbosity=2)
# pymanopt_solver = SteepestDescent(maxiter=max_iters, mingradnorm=mingradnorm, logverbosity=2)

######## DATA ########
N = 1
dx = 784
dy = 625
# dx = 225
# dy = 81
nx = [4,7,4,7]
ny = [5,5,5,5]
# nx = [3, 5, 3, 5]
# ny = [3, 3, 3, 3]

# r = [1, 20, 35, 35, 1]
r = [1, 9, 9, 9, 1]

n = map(lambda x,y:x*y, nx, ny)

X_data = np.random.uniform(size=[dx,N]).astype('float32')
W_gt = np.random.uniform(size=[dy,dx]).astype('float32')
Y_data = np.matmul(W_gt, X_data)# + 0.01*np.random.randn(N,dy)

X = tf.placeholder(tf.float32, [dx, N])
Y = tf.placeholder(tf.float32, [dy, N])

# random init seed
#np.random.seed(28980)
np.random.seed(89431896)



#### Standard TF GD ####
print('Starting TF GD...')
#gdlr = 1e-5
initializer = tf.glorot_uniform_initializer()
W_tf_gd = tf.get_variable('W_tf_gd', shape=[dy, dx], initializer=initializer)

cost_tf_gd = tf.reduce_mean(0.5*tf.square(Y - tf.matmul(W_tf_gd, X)))
# grad_tf_gd = tf.gradients(cost_tf_gd, W_tf_gd)
# norm_tf_gd = tf.norm(grad_tf_gd)
tf_gd_solver = tf.train.AdamOptimizer().minimize(cost_tf_gd)
#tf_gd_solver = tf.train.GradientDescentOptimizer(learning_rate=gdlr).minimize(cost_tf_gd)

sess.run(tf.global_variables_initializer())

tfgd_loss = []
# while(sess.run(tf.less(mingradnorm, norm_tf_gd), feed_dict={X: X_data, Y: Y_data})):
while(1):
	_, tmp = sess.run([tf_gd_solver, cost_tf_gd], feed_dict={X: X_data, Y: Y_data})
	tfgd_loss.append(tmp)
	print(tmp)
	if tmp < mincost:
		break



#### T3F GD ####
print('Starting T3F GD...')
#t3fgdlr = 1e-5
# Using basics of tensorflow just do the autodiff on the whole thing
initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
W_t3f_gd = t3f.get_variable('W_t3f_gd', initializer=initializer)

cost_t3f_gd = tf.reduce_mean(0.5*tf.square(Y - t3f.matmul(W_t3f_gd, X)))
# grad_t3f_gd = tf.gradients(cost_t3f_gd, list(W_t3f_gd.tt_cores))
# norm_t3f_gd = t3f.frobenius_norm(t3f.TensorTrain(grad_t3f_gd), epsilon=1e-10)
t3f_gd_solver = tf.train.AdamOptimizer().minimize(cost_t3f_gd)
#t3f_gd_solver = tf.train.GradientDescentOptimizer(learning_rate=t3fgdlr).minimize(cost_t3f_gd)

sess.run(tf.global_variables_initializer())

t3fgd_loss = []
# while(sess.run(tf.less(mingradnorm, norm_t3f_gd), feed_dict={X: X_data, Y: Y_data})):
while(1):
    _, tmp = sess.run([t3f_gd_solver, cost_t3f_gd], feed_dict={X: X_data, Y: Y_data})
    t3fgd_loss.append(tmp)
    print(tmp)
    if tmp < mincost:
		break



#### T3F RGD ####
print('Starting T3F RGD...')
rgdlr = 1e-3
# using riemannian projection implemented by t3f, compute a separate update
# this requires projection/rounding which should be computationally intensive
initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
W_t3f_rgd = t3f.get_variable('W_t3f_rgd', initializer=initializer)

cost_t3f_rgd = tf.reduce_mean(0.5*tf.square(Y - t3f.matmul(W_t3f_rgd, X)))

# least squares derivative
grad = t3f.to_tt_matrix( tf.matmul((Y - t3f.matmul(W_t3f_rgd, X)), -1*tf.transpose(X) ), shape=[ny, nx], max_tt_rank=max(r) )
riemannian_grad = t3f.riemannian.project(grad, W_t3f_rgd)
# norm_t3f_rgd = t3f.frobenius_norm(riemannian_grad, epsilon=1e-10)

train_step = t3f.assign(W_t3f_rgd, t3f.round(W_t3f_rgd - rgdlr * riemannian_grad, max_tt_rank=max(r)) )

sess.run(tf.global_variables_initializer())

t3frgd_loss = []
# while(sess.run(tf.less(mingradnorm, norm_t3f_rgd), feed_dict={X: X_data, Y: Y_data})):
while(1):
	_, tmp = sess.run([train_step.op, cost_t3f_rgd], feed_dict={X: X_data, Y: Y_data})
	t3frgd_loss.append(tmp)
	# print(sess.run(norm_t3f_rgd, feed_dict={X: X_data, Y: Y_data}))
	print(tmp)
	if tmp < mincost:
		break



# #### Own TT GD ####
# #owngdlr = 1e-2
# # my simple implementation of TT, just for simple comparison. using autodiff tensorflow
# W_own_gd = TTtfVariable(shape=[ny,nx], r=r)

# cost_own_gd = tf.reduce_mean(0.5*tf.square(Y_data - tf.matmul(W_own_gd.getW(), X_data)))
# grad_own_gd = tf.gradients(cost_own_gd, W_own_gd)
# norm_own_gd = tf.norm(grad_own_gd)
# tt_own_solver = tf.train.AdamOptimizer().minimize(cost_own_gd)
# #tt_own_solver = tf.train.GradientDescentOptimizer(learning_rate=owngdlr).minimize(cost_own_gd)

# sess.run(tf.global_variables_initializer())

# ttowngd_loss = []
# while(sess.run(tf.less(mingradnorm, norm_own_gd), feed_dict={X: X_data, Y: Y_data})):
#     _, tmp = sess.run([tt_own_solver, cost_own_gd], feed_dict={X: X_data, Y: Y_data})
#     ttowngd_loss.append(tmp)



# #### Own EOTT GD ####
# # using pymanopt
print('Starting EOTT...')
W_EOTT_gd = OTTtfVariable(shape=[ny,nx], r=r)

cost_eott_gd = tf.reduce_mean(0.5*tf.square(Y_data - tf.matmul(W_EOTT_gd.getW(), X_data)))

eott_problem = Problem(manifold=Product(W_EOTT_gd.getManifoldList()), cost=cost_eott_gd, arg=W_EOTT_gd.getQ())

_, eott_log = pymanopt_solver.solve(eott_problem)
eott_loss = eott_log['iterations']['f(x)']



# #### Own AOTT GD ####
# # using pymanopt
print('Starting AOTT...')
W_AOTT_gd = aOTTtfVariable(shape=[ny,nx], r=r)

cost_aott_gd = tf.reduce_mean(0.5*tf.square(Y_data - tf.matmul(W_AOTT_gd.getW(), X_data)))

aott_problem = Problem(manifold=Product(W_AOTT_gd.getManifoldList()), cost=cost_aott_gd, arg=W_AOTT_gd.getQ())

_, aott_log = pymanopt_solver.solve(aott_problem)
aott_loss = aott_log['iterations']['f(x)']


#### Own EOTT GD (with Rudra Lifting) ####
## TODO ##



#### Own AOTT GD (with Rudra Lifting) ####
## TODO ##



## plot data
fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.set_yscale('log')
ax.plot(np.arange(1,len(tfgd_loss)+1,1), tfgd_loss, 'k:', label='noTT-ADAM')
ax.plot(np.arange(1,len(t3fgd_loss)+1,1), t3fgd_loss, 'r-', label='T3F-ADAM')
ax.plot(np.arange(1,len(t3frgd_loss)+1,1), t3frgd_loss, 'b-', label='T3F-RGD')
# ax.plot(np.arange(1,len(ttowngd_loss)+1,1), ttowngd_loss, 'g-', label='myTT')
ax.plot(np.arange(1,len(eott_loss)+1,1), eott_loss, 'k-', label='eOTT-SDLS')
ax.plot(np.arange(1,len(aott_loss)+1,1), aott_loss, 'm-', label='aOTT-SDLS')
plt.legend()
plt.show()