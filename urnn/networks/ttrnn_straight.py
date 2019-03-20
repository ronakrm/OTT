import numpy as np
import tensorflow as tf

from .tensor_rnn_cell import mySimpleRNNCell

###################################################################################################333

class TTRNNCell(mySimpleRNNCell):

    def __init__(self, input_shape, hidden_shape, output_shape, activation, ttRank, ttvar, reuse=None):
        super(TTRNNCell, self).__init__(input_shape, hidden_shape, output_shape, activation, reuse=reuse)
        self.ttvar = ttvar
        self.maxTTrank = ttRank

        self.setup_inp_to_h()
        self.setup_h_to_h()
        self.setup_h_to_out()

    def input_layer(self, inputs):
        inputs_mul = tf.transpose(self.w_ih.mult(tf.transpose(inputs)))
        return tf.reshape(inputs_mul, [-1, self.state_size])

    def hidden_layer(self, hidden_state):
        tmp = tf.transpose(self.w_hh.mult(tf.transpose(hidden_state)))
        return tf.reshape(tmp, [-1,self.state_size])

    def output_layer(self, new_state):
        output = tf.transpose(self.w_ho.mult(tf.transpose(new_state))) + self.b_o
        return output

    def setup_inp_to_h(self):
        if self.input_shape is not None:
            self.w_ih = self.ttvar(name="w_ih", shape=[self.state_shape, self.input_shape], r=self.maxTTrank)
        else:
            self.w_ih = tf.get_variable("w_ih", shape=[self.state_size, self.input_size], 
                            initializer=tf.contrib.layers.xavier_initializer())    

        self.b_h = tf.Variable(tf.zeros(self.state_size), name="b_h")
        
    def setup_h_to_h(self):
        self.w_hh = self.ttvar(name="w_hh", shape=[self.state_shape,self.state_shape], r=self.maxTTrank)

    def setup_h_to_out(self):
        self.w_ho = self.ttvar(name="w_ho", shape=[self.output_shape, self.state_shape], r=self.maxTTrank)
        self.b_o = tf.Variable(tf.zeros(self.output_size), name="b_h")