import tensorflow as tf
from problems.movingmnist import MovingMnistProblemDataset
from networks.tf_rnn import TFRNN
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

glob_learning_rate = 0.001*10
glob_decay = 0.9

def serialize_loss(loss, name):
    file=open(loss_path + name, 'w')
    for l in loss:
        file.write("{0}\n".format(l))

class Main:
    def init_data(self):
        print('Generating data...')

        self.mm_batch_size = 32 #5, 10, 32
        self.mm_epochs = 10 #10, 50, 100
        self.nsamps = 10000
        self.seqlen = 3

        # # 64 EXP
        # self.hidden_size = 64
        # self.nh = [2, 4, 4, 2]
        # self.frame_size = 64
        # self.nx = [4, 16, 16, 4]
        # self.ttRank = 64
        # self.digit_size = 28
        # self.speed = 5

        # # 512 EXP
        # self.hidden_size = 4096
        # self.nh = [4, 4, 4, 4, 4,4]
        # frame_size = 512
        # self.nx = [4,8,16,16,8,4]
        # self.ttRank = 64
        # self.digit_size = 112
        # self.speed = 25

        # 1024 EXP
        self.hidden_size = 4096
        self.nh = [4, 4, 4, 4, 4,4]
        frame_size = 1024
        self.nx = [4,16,16,16,16,4]
        self.ttRank = 64
        self.digit_size = 224
        self.speed = 125


        self.vec_size = self.frame_size**2
        self.mm_data= MovingMnistProblemDataset(self.nsamps, self.seqlen,
                                num_sz=self.digit_size,
                                frame_size=self.frame_size,speed=self.speed)

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=',sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        serialize_loss(net.get_loss_list(), net.name + sample_len)
        print('Training network ', net.name, ' done.')


    def train_ottrnn(self):
        print('Initializing and training OTTRNN...')

        # moving mnist 
        tf.reset_default_graph()
        self.mm_ottrnn = TFRNN(
            name = "mm_ottrnn",
            num_in = self.vec_size,
            num_hidden = self.hidden_size,
            num_out = self.vec_size,
            num_target = self.vec_size,
            single_output = False,
            rnn_cell = OTTRNNCell,
            activation_hidden = tf.tanh, # modReLU
            activation_out = tf.identity,
            optimizer = tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function = tf.nn.sigmoid_cross_entropy_with_logits,
            seq_len = self.seqlen,
            ttRank = self.ttRank,
            imTTmodes = self.nx,
            hdTTmodes = self.nh)
        self.train_network(self.mm_ottrnn, self.mm_data, 
                           self.mm_batch_size, self.mm_epochs)

        print('Init and training OTTRNN done.')


    def train_networks(self):
        print('Starting training...')

        # MOVING MNIST
        main.train_ottrnn()
        
        
        print('Done and done.')

main=Main()
main.init_data()
main.train_networks()
