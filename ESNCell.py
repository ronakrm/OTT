import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import 
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

class CapRNNcell(tf.contrib.rnn.RNNCell):
    def __init__(self, input_dim):
        self.input_dim = input_dim

        self.W = tf.get_variable("W", [self.input_dim , 1], tf.float32)
        self.b = tf.get_variable("b", [1])

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state):
        output =state*tf.nn.sigmoid(tf.matmul(inputs, self.W)+ self.b)

        return output, output

def CapRnnModel(timeSeries_before_forgetting_gate, init_cap):

    cap_cell = CapRNNcell(input_dim=3)
    cap_series, final_cap = tf.nn.dynamic_rnn(cell=cap_cell, inputs=timeSeries_before_forgetting_gate, initial_state=init_cap)

    return  cap_series , final_cap

x_place=tf.placeholder(tf.float32 , [1,2,3])
init_cap_place=tf.placeholder(tf.float32 , [1,1])

y=CapRnnModel(x_place, init_cap_place)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    a=np.random.rand(1,2,3)
    b=np.random.rand(1,1)
    result=sess.run(y,feed_dict={x_place:a , init_cap_place:b})
    print(result)