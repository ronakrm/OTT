import numpy as np
import tensorflow as tf

from .tensor_rnn_cell import mySimpleRNNCell

###################################################################################################333

class myRNNCell(mySimpleRNNCell):

    def __init__(self, input_shape, hidden_shape, output_shape, activation, reuse=None):
        super(myRNNCell, self).__init__(input_shape, hidden_shape, output_shape, activation, reuse=reuse)

        self.setup_inp_to_h()
        self.setup_h_to_h()
        self.setup_h_to_out()

    def input_layer(self, inputs):
        return tf.matmul(inputs, self.w_ih)

    def hidden_layer(self, hidden_state):
        return tf.matmul(hidden_state, self.w_hh)

    def output_layer(self, new_state):
        return tf.matmul(new_state, self.w_ho) + self.b_o

    def setup_inp_to_h(self):
        self.w_ih = tf.get_variable("w_ih", shape=[self.input_size, self.state_size], 
                            initializer=tf.contrib.layers.xavier_initializer())

        self.b_h = tf.Variable(tf.zeros(self.state_size), name="b_h")
        
    def setup_h_to_h(self):
        self.w_hh = tf.get_variable("w_hh", shape=[self.state_size, self.state_size], 
                            initializer=tf.contrib.layers.xavier_initializer())

    def setup_h_to_out(self):
        self.w_ho = tf.get_variable("w_ho", shape=[self.state_size, self.output_size], 
                            initializer=tf.contrib.layers.xavier_initializer())
        self.b_o = tf.Variable(tf.zeros(self.output_size), name="b_h")