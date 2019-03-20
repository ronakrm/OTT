import numpy as np
import tensorflow as tf

from .tensor_rnn_cell import mySimpleRNNCell


###################################################################################################333

class CNN3DRNNCell(mySimpleRNNCell):

    def __init__(self, input_shape, hidden_shape, output_shape, activation, ttRank, ttvar, reuse=None):
        super(CNN3DRNNCell, self).__init__(input_shape, hidden_shape, output_shape, activation, reuse=reuse)
        # save class variables
        self.ttvar = ttvar
        self.maxTTrank = ttRank

        self.setup_inp_to_h()
        self.setup_h_to_h()
        self.setup_h_to_out()


    # def input_layer(self, inputs):
        # inp = tf.reshape(inputs, [-1, 121*145*121])
        # inp = tf.transpose(self.w_ih.mult(tf.transpose(inputs)))
        # return tf.reshape(tf.matmul(inp, self.w_ih), [-1, 4, 5, 4, 1]) 
        # return tf.matmul(inp, self.w_ih)
    def input_layer(self, inputs):
        # return tf.transpose(self.w_ih.mult(tf.transpose(inputs)))
        tmp = [None]*(self.inp_convlayers+1)
        tmp[0] = tf.reshape(inputs, [-1, 121, 145, 121, 1])
        for i in range(0,self.inp_convlayers):
            tmp[i+1] = tf.nn.conv3d(tmp[i], filter=self.inp_filters[i],
                                    strides=self.inp_strides[i],
                                    padding="SAME", name=self.inp_names[i])
        return tf.reshape(tmp[self.inp_convlayers], [-1, self.state_size])

    def hidden_layer(self, hidden_state):
        # return tf.matmul(hidden_state, self.w_hh)
        # new_state = tf.nn.conv3d(hidden_state, filter=self.hid_filter,
        #                             strides=[1,1,1,1,1],
        #                             padding="SAME", name=self.hid_name)
        return tf.transpose(self.w_hh.mult(tf.transpose(hidden_state)))
        # return new_state

    def output_layer(self, new_state):
        # # hid = tf.reshape(new_state, [-1, 4*5*4])
        # # inp = tf.transpose(self.w_ih.mult(tf.transpose(inputs)))
        # return tf.matmul(new_state, self.w_ho)
        tmp = [None]*(self.out_convlayers+1)
        tmp[0] = tf.reshape(new_state, [-1, 4, 5, 4, 128])
        for i in range(0,self.out_convlayers):
            tmp[i+1] = tf.nn.conv3d_transpose(tmp[i], filter=self.out_filters[i],
                    output_shape=[tf.shape(new_state)[0]]+self.out_shapes[i], strides=self.out_strides[i],
                    padding="SAME", name=self.out_names[i])

        return tf.reshape(tmp[self.out_convlayers], [-1, 121*145*121])


    def setup_inp_to_h(self):
        # self.w_ih = tf.get_variable("w_ih", shape=[self.input_size, self.state_size], 
        #                     initializer=tf.contrib.layers.xavier_initializer())

        # self.w_ih = self.ttvar(name="w_ih", shape=[self._state_shape, self._input_shape+[1]], r=self.maxTTrank)

        self.b_h = tf.Variable(tf.zeros(self.state_size), name="b_h")

    # def setup_inp_to_h(self):

        # set up input -> hidden connection
        # if self.input_shape is not None:
        # self.w_ih = self.ttvar(name="w_ih", shape=[self._state_shape, self._input_shape], r=self.maxTTrank)
        # else:
            # self.w_ih = tf.get_variable("w_ih", shape=[self._num_units, self._num_in], 
                            # initializer=tf.contrib.layers.xavier_initializer())    
               # 

        self.inp_convlayers = 5
        # setup output filters
        self.inp_strides = self.inp_convlayers*[None]
        self.inp_names = self.inp_convlayers*[None]
        self.inp_filtnames = self.inp_convlayers*[None]
        self.inp_filters = self.inp_convlayers*[None]
        self.inp_filtshapes = [ [3,3,3,1,4],
                                [3,3,3,4,16],
                                [3,3,3,16,64],
                                [3,3,3,64,256],
                                [3,3,3,256,128] ]

        for i in range(0, self.inp_convlayers):
            self.inp_strides[i] = [1,2,2,2,1]
            self.inp_names[i] = "inp_conv3d_" + str(i)
            self.inp_filtnames[i] = "inp_filt_" + str(i)
            self.inp_filters[i] = tf.get_variable(self.inp_filtnames[i], shape=self.inp_filtshapes[i],
                               initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=None,
                               trainable=True)

        # self.b_h = tf.Variable(tf.zeros(self._state_shape), name="b_h")
        
        

    def setup_h_to_h(self):
        # self.w_hh = tf.get_variable("w_hh", shape=[self.state_size, self.state_size], 
                            # initializer=tf.contrib.layers.xavier_initializer())
        # self.hid_filtshape = [1,1,1,1,1]
        # self.hid_name = "hid_conv3d"
        # self.hid_filtname = "hid_filt"
        # self.hid_filter = tf.get_variable(self.hid_filtname, shape=self.hid_filtshape,
                               # initializer=tf.contrib.layers.xavier_initializer(),
                               # regularizer=None,
                               # trainable=True)
        self.w_hh = self.ttvar(name="w_hh", shape=[self._state_shape, self._state_shape], r=self.maxTTrank)

    def setup_h_to_out(self):
        # self.w_ho = tf.get_variable("w_ho", shape=[self.state_size, self.output_size], 
        #                     initializer=tf.contrib.layers.xavier_initializer())
        self.out_convlayers = 5
        # setup output filters
        self.out_strides = self.out_convlayers*[None]
        self.out_names = self.out_convlayers*[None]
        self.out_filtnames = self.out_convlayers*[None]
        self.out_filters = self.out_convlayers*[None]
        self.out_shapes = [ [8,10,8,256],
                            [16,19,16,64],
                            [31,37,31,16],
                            [61,73,61,4],
                            [121,145,121,1] ]
        self.out_filtshapes = [ [3,3,3,256,128],
                                [3,3,3,64,256],
                                [3,3,3,16,64],
                                [3,3,3,4,16],
                                [3,3,3,1,4] ]

        for i in range(0, self.out_convlayers):
            self.out_strides[i] = [1,2,2,2,1]
            self.out_names[i] = "out_conv3d_" + str(i)
            self.out_filtnames[i] = "out_filt_" + str(i)
            self.out_filters[i] = tf.get_variable(self.out_filtnames[i], shape=self.out_filtshapes[i],
                               initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=None,
                               trainable=True)