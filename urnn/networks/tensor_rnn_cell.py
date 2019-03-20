import numpy as np
import tensorflow as tf

###################################################################################################333

class mySimpleRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, input_shape, hidden_shape, output_shape, activation, reuse=None):
        super(mySimpleRNNCell, self).__init__(_reuse=reuse)
        # save class variables
        self._input_shape = input_shape
        self._input_size = int(np.prod(input_shape))
        self._state_shape = hidden_shape
        self._state_size = int(np.prod(hidden_shape))
        self._output_shape = output_shape
        self._output_size = int(np.prod(output_shape))
        self.activation = activation

    # needed properties
    @property
    def input_shape(self):
        return self._input_shape # real

    @property
    def input_size(self):
        return self._input_size # real

    @property
    def state_shape(self):
        return self._state_shape # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_shape(self):
        return self._output_shape # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, state):

        # compute input-hidden
        i_to_h = self.input_layer(inputs)

        # compute hidden-hidden
        h_to_h = self.hidden_layer(state)

        # compute preactivation
        preact = i_to_h + h_to_h + self.b_h

        # compute next state
        new_state = self.activation(preact)

        # compute hidden-output
        output = self.output_layer(new_state)

        return output, new_state


    def input_layer(self, inputs):
        return NotImplementedError

    def hidden_layer(self, hidden_state):
        return NotImplementedError

    def output_layer(self, new_state):
        return NotImplementedError

    def setup_inp_to_h():
        return NotImplementedError

    def setup_h_to_h():
        return NotImplementedError

    def setup_h_to_out():
        return NotImplementedError
