## Compare all things
## least squares

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
approx_error = 1e-10
max_iters = 1000
sess = tf.Session()

gdlr = 1e-4
rgdlr = 1e-8
owngdlr = 1e-12
pymanopt_solver = SteepestDescent(maxiter=max_iters, logverbosity=2)

######## DATA ########
N = 1
dx = 784
dy = 625

X_data = np.random.uniform(size=[dx,N]).astype('float32')
W_gt = np.random.uniform(size=[dy,dx]).astype('float32')
Y_data = np.matmul(W_gt, X_data)# + 0.01*np.random.randn(N,dy)

nx = [4,7,4,7]
ny = [5,5,5,5]
n = map(lambda x,y:x*y, nx, ny)
r = [1, 20, 35, 35, 1]

X = tf.placeholder(tf.float32, [dx, N])
Y = tf.placeholder(tf.float32, [dy, N])


# random init seed
np.random.seed(28980)


#### Standard TF GD ####
initializer = tf.glorot_uniform_initializer()
W_tf_gd = tf.get_variable('W_tf_gd', shape=[dy, dx], initializer=initializer)

cost_tf_gd = tf.reduce_mean(0.5*tf.square(Y - tf.matmul(W_tf_gd, X)))
tf_gd_solver = tf.train.GradientDescentOptimizer(learning_rate=gdlr).minimize(cost_tf_gd)

sess.run(tf.global_variables_initializer())

tfgd_loss = []
for it in range(max_iters):
    _, tmp = sess.run([tf_gd_solver, cost_tf_gd], feed_dict={X: X_data, Y: Y_data})
    tfgd_loss.append(tmp)


#### T3F GD ####
# Using basics of tensorflow just do the autodiff on the whole thing
initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
W_t3f_gd = t3f.get_variable('W_t3f_gd', initializer=initializer)

cost_t3f_gd = tf.reduce_mean(0.5*tf.square(Y - t3f.matmul(W_t3f_gd, X)))
t3f_gd_solver = tf.train.GradientDescentOptimizer(learning_rate=gdlr).minimize(cost_t3f_gd)

sess.run(tf.global_variables_initializer())

t3fgd_loss = []
for it in range(max_iters):
    _, tmp = sess.run([t3f_gd_solver, cost_t3f_gd], feed_dict={X: X_data, Y: Y_data})
    t3fgd_loss.append(tmp)



#### T3F RGD ####
# using riemannian projection implemented by t3f, compute a separate update
# this requires projection/rounding which should be computationally intensive
initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
W_t3f_rgd = t3f.get_variable('W_t3f_rgd', initializer=initializer)

cost_t3f_rgd = tf.reduce_mean(0.5*tf.square(Y - t3f.matmul(W_t3f_rgd, X)))

# least squares derivative
grad = t3f.to_tt_matrix( tf.matmul((Y - t3f.matmul(W_t3f_rgd, X)), tf.transpose(X) ), shape=[ny, nx], max_tt_rank=max(r) )
riemannian_grad = t3f.riemannian.project(grad, W_t3f_rgd)

train_step = t3f.assign(W_t3f_rgd, t3f.round(W_t3f_rgd - rgdlr * riemannian_grad, max_tt_rank=max(r)) )

sess.run(tf.global_variables_initializer())

t3frgd_loss = []
for i in range(max_iters):
	_, tmp = sess.run([train_step.op, cost_t3f_rgd], feed_dict={X: X_data, Y: Y_data})
	t3frgd_loss.append(tmp)



#### Own TT GD ####
# my simple implementation of TT, just for simple comparison. using autodiff tensorflow
W_own_gd = TTtfVariable(shape=[ny,nx], r=r)

cost_own_gd = tf.reduce_mean(0.5*tf.square(Y_data - tf.matmul(W_own_gd.getW(), X_data)))
tt_own_solver = tf.train.GradientDescentOptimizer(learning_rate=owngdlr).minimize(cost_own_gd)

sess.run(tf.global_variables_initializer())

ttowngd_loss = []
for it in range(max_iters):
    _, tmp = sess.run([tt_own_solver, cost_own_gd], feed_dict={X: X_data, Y: Y_data})
    ttowngd_loss.append(tmp)



#### Own EOTT GD ####
# using pymanopt
W_EOTT_gd = OTTtfVariable(shape=[ny,nx], r=r)

cost_eott_gd = tf.reduce_mean(0.5*tf.square(Y_data - tf.matmul(W_EOTT_gd.getW(), X_data)))

eott_problem = Problem(manifold=Product(W_EOTT_gd.getManifoldList()), cost=cost_eott_gd, arg=W_EOTT_gd.getQ())

_, eott_log = pymanopt_solver.solve(eott_problem)

eott_loss = eott_log['iterations']['f(x)']


#### Own AOTT GD ####
# using pymanopt
W_AOTT_gd = aOTTtfVariable(shape=[ny,nx], r=r)

cost_aott_gd = tf.reduce_mean(0.5*tf.square(Y_data - tf.matmul(W_AOTT_gd.getW(), X_data)))

aott_problem = Problem(manifold=Product(W_AOTT_gd.getManifoldList()), cost=cost_aott_gd, arg=W_AOTT_gd.getQ())

pymanopt_solver.solve(aott_problem)

_, aott_log = pymanopt_solver.solve(aott_problem)
aott_loss = aott_log['iterations']['f(x)']


#### Own EOTT GD (with Rudra Lifting) ####
## TODO ##



#### Own AOTT GD (with Rudra Lifting) ####
## TODO ##



## collect data
# wall clock
# GD iterations
# loss
fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.set_yscale('log')
xax = np.arange(1,max_iters+1,1)
ax.plot(xax, tfgd_loss, 'k:', label='noTT')
ax.plot(xax, t3fgd_loss, 'r-', label='T3FGD')
ax.plot(xax, t3frgd_loss, 'b-', label='T3FRGD')
ax.plot(xax, ttowngd_loss, 'g-', label='myTT')
ax.plot(xax, eott_loss, 'k-', label='eOTT')
ax.plot(xax, aott_loss, 'm-', label='aOTT')
plt.legend()
plt.show()