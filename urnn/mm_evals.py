import tensorflow as tf
from problems.movingmnist import MovingMnistProblemDataset
from problems.mnist import MnistProblemDataset
from networks.tf_rnn import TFRNN
from networks.urnn_cell import URNNCell
from networks.ottrnn_cell import OTTRNNCell
import numpy as np

'''
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function):
'''

loss_path='results/'

glob_learning_rate = 0.001
glob_decay = 0.9

def serialize_loss(loss, name):
    file=open(loss_path + name, 'w')
    for l in loss:
        file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        self.mm_batch_size = 3 #10, 32
        self.mm_epochs = 5 #10, 50, 100

        ## THIS NEEDS TO ALSO BE UPDATED IN TF_RNN and OTTRNN_CELL
        ## SHOULD UPDATE TT DECOMP MODES TO MATCH 
        self.hidden_size = 4096 # 64, 256, 1024, 4096
        frame_size = 256   #1024   # default 64

        self.vec_size = frame_size**2

        digit_size = 56   #224    # default 28
        self.mm_data= MovingMnistProblemDataset(200, 3, num_sz=digit_size, frame_size=frame_size)

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=',sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        serialize_loss(net.get_loss_list(), net.name + sample_len)
        print('Training network ', net.name, ' done.')

    # def train_urnn_for_timestep_idx(self, idx):
    #     print('Initializing and training URNNs for one timestep...')

    #     # moving mnist 
    #     tf.reset_default_graph()
    #     self.mm_urnn=TFRNN(
    #         name="mm_urnn",
    #         num_in=4096,
    #         num_hidden=1024,
    #         num_out=4096,
    #         num_target=4096,
    #         single_output=False,
    #         rnn_cell=URNNCell,
    #         activation_hidden=None, # modReLU
    #         activation_out=tf.identity,
    #         optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
    #         loss_function=tf.nn.sigmoid_cross_entropy_with_logits)
    #     self.train_network(self.mm_urnn, self.mm_data, 
    #                        self.mm_batch_size, self.mm_epochs)

    #     print('Init and training URNNs for one timestep done.')


    def train_ottrnn_for_timestep_idx(self, idx):
        print('Initializing and training OTTRNNs for one timestep...')

        # moving mnist 
        tf.reset_default_graph()
        self.mm_ottrnn=TFRNN(
            name="mm_ottrnn",
            num_in=self.vec_size,
            num_hidden=self.hidden_size,
            num_out=self.vec_size,
            num_target=self.vec_size,
            single_output=False,
            rnn_cell=OTTRNNCell,
            activation_hidden=tf.tanh, # modReLU
            activation_out=tf.identity,
            #optimizer=tf.train.GradientDescentOptimizer(learning_rate=glob_learning_rate),
            optimizer=tf.train.RMSPropOptimizer(learning_rate=10*glob_learning_rate, decay=glob_decay),
            loss_function=tf.nn.sigmoid_cross_entropy_with_logits)
        self.train_network(self.mm_ottrnn, self.mm_data, 
                           self.mm_batch_size, self.mm_epochs)

        print('Init and training URNNs for one timestep done.')


    # def train_rnn_lstm_for_timestep_idx(self, idx):
    #     print('Initializing and training RNN&LSTM for one timestep...')

    #     tf.reset_default_graph()
    #     self.mm_simple_rnn=TFRNN(
    #         name="mm_simple_rnn",
    #         num_in=4096,
    #         num_hidden=4096,
    #         num_out=4096,
    #         num_target=4096,
    #         single_output=False,
    #         rnn_cell=tf.contrib.rnn.BasicRNNCell,
    #         activation_hidden=tf.tanh,
    #         activation_out=tf.identity,
    #         optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
    #         loss_function=tf.nn.sigmoid_cross_entropy_with_logits)
    #     self.train_network(self.mm_simple_rnn, self.mm_data, 
    #                        self.mm_batch_size, self.mm_epochs)

    #     tf.reset_default_graph()
    #     self.mm_lstm=TFRNN(
    #         name="mm_lstm",
    #         num_in=4096,
    #         num_hidden=1024,
    #         num_out=4096,
    #         num_target=4096,
    #         single_output=False,
    #         rnn_cell=tf.contrib.rnn.LSTMCell,
    #         activation_hidden=tf.tanh,
    #         activation_out=tf.identity,
    #         optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
    #         loss_function=tf.nn.sigmoid_cross_entropy_with_logits)
    #     self.train_network(self.mm_lstm, self.mm_data, 
    #                        self.mm_batch_size, self.mm_epochs)

    #     print('Init and training networks for one timestep done.')

    def train_networks(self):
        print('Starting training...')

        # MOVING MNIST
        main.train_ottrnn_for_timestep_idx(0)
        # main.train_rnn_lstm_for_timestep_idx(0)
        # main.train_urnn_for_timestep_idx(0)
        
        
        print('Done and done.')

main=Main()
main.init_data()
main.train_networks()
