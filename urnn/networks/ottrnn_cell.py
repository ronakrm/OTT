import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from vars.sOTTtfVariable import sOTTtfVariable
# from vars.aOTTtfVariable import aOTTtfVariable
from vars.TTtfVariable import TTtfVariable
# import t3f


###################################################################################################333

# 4k / 7k trainable params
class OTTRNNCell(tf.contrib.rnn.RNNCell):
    """The most basic URNN cell.
    Args:
        num_units (int): The number of units in the LSTM cell, hidden layer size.
        num_in: Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in, nh, nx, ttRank, reuse=None):
        super(OTTRNNCell, self).__init__(_reuse=reuse)
        # save class variables
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units

        ## EDIT THIS TO REFLECT INPUT AND HIDDEN DIM SIZES
        maxTTrank = ttRank
        # nx = [4,16,16,16,16,4] #prod statesize
        # nx = [4, 8, 8, 8, 8, 4]
        # nh = [4, 4, 4, 4, 4, 4] #prod statesize
        

        # set up input -> hidden connection
        self.w_ih = TTtfVariable(name="w_ih", shape=[nh,nx], r=maxTTrank)
        # self.w_ih = tf.get_variable("w_ih", shape=[num_units, num_in], 
                                    # initializer=tf.contrib.layers.xavier_initializer())
        self.b_h = tf.Variable(tf.zeros(self._state_size), # state size actually
                                    name="b_h")
        
        self.W1 = sOTTtfVariable(name="W1", shape=[nh,nh], r=maxTTrank)
        # self.W1 = TTtfVariable(name="W1", shape=[nh,nh], r=maxTTrank)
        # initializer = t3f.glorot_initializer([ny, nx], tt_rank=r)
        # self.W1 = t3f.get_variable('W1', initializer=initializer) 
        # self.W1 = tf.get_variable(name="W1", shape=[num_units, num_units],
                                # initializer=tf.contrib.layers.xavier_initializer())
        # print(maxTTrank)

    # needed properties
    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """
        #print("cell.call inputs:", inputs.shape, inputs.dtype)
        #print("cell.call state:", state.shape, state.dtype)

        # prepare input linear combination
        # inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih)) # [batch_sz, 2*num_units]
        inputs_mul = tf.transpose(self.w_ih.mult(tf.transpose(inputs)))
        # [batch_sz, num_units]

        state_mul = tf.transpose(self.W1.mult(tf.transpose(state)))
        # state_mul = t3f.matmul(state, self.W1)
        # state_mul = tf.matmul(state, self.W1)

        # [batch_sz, num_units]
        
        # calculate preactivation
        preact = inputs_mul + state_mul + self.b_h
        # [batch_sz, num_units]

        new_state = tf.nn.relu(preact) # [batch_sz, num_units] C
        #new_state = tf.concat([tf.real(new_state_c), tf.imag(new_state_c)], 1) # [batch_sz, 2*num_units] R
        # outside network (last dense layer) is ready for 2*num_units -> num_out
        output = new_state
        # print("cell.call output:", output.shape, output.dtype)
        # print("cell.call new_state:", new_state.shape, new_state.dtype)

        return output, new_state

