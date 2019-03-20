import argparse
import numpy as np
import ast
import tensorflow as tf
# from problems.s_movingmnist import MovingMnistProblemDataset
from problems.movingmnist import MovingMnistProblemDataset
from networks.tfrnn import TFRNN
# from networks.ottrnn_cell import OTTRNNCell as mycell
from networks.ttrnn_straight import TTRNNCell as mycell
# from networks.rnn_straight import myRNNCell as mycell
from vars.sOTTtfVariable import sOTTtfVariable as mytt
# from vars.TTtfVariable import TTtfVariable as mytt

import sys
sys.path.insert(0, '../')
from seqVisualizer import seqVisualizer

glob_learning_rate = 0.001
glob_decay = 0.9

class Main:
    def init_data(self, args):
        print('Generating data...')

        self.mm_batch_size = args.batch_size
        self.mm_epochs = args.epochs
        self.nsamps = args.nsamps
        self.seqlen = args.seqlen
        self.hidden_size = args.sh
        self.nh = ast.literal_eval(args.nh)
        self.frame_size = args.frame_size
        self.nx = ast.literal_eval(args.nx)
        self.ttRank = int(args.ttRank)
        self.digit_size = int(args.digit_size)
        self.speed = int(args.speed)


        self.datapath = 'data/'+str(self.seqlen)+'_'+str(self.frame_size) \
                        +'_'+str(self.digit_size)+'_'+str(self.speed)+'/'

        self.vec_size = self.frame_size**2

        assert(np.prod(self.nh)==self.hidden_size)
        assert(np.prod(self.nx)==self.vec_size)

        self.mm_data= MovingMnistProblemDataset(self.nsamps, self.seqlen,
                                datapath=self.datapath, num_sz=self.digit_size,
                                frame_size=self.frame_size,speed=self.speed)

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        print('Training network ', net.name, '... timesteps=',self.seqlen)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        print('Training network ', net.name, ' done.')


    def train_ottrnn(self):
        print('Initializing and training OTTRNN...')

        viz = seqVisualizer(batch_size=self.mm_batch_size, seqlen=self.seqlen, frame_size=np.array([self.frame_size, self.frame_size]))

        # moving mnist 
        tf.reset_default_graph()
        self.mm_ottrnn = TFRNN(
            name = "mm_ottrnn",
            num_in = self.vec_size,
            num_hidden = self.hidden_size,
            num_out = self.vec_size,
            num_target = self.vec_size,
            single_output = False,
            rnn_cell = mycell,
            activation_hidden = tf.nn.relu, # modReLU
            activation_out = tf.nn.sigmoid,
            optimizer = tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function = tf.nn.sigmoid_cross_entropy_with_logits,
            seq_len = self.seqlen,
            ttRank = self.ttRank,
            imTTmodes = self.nx,
            hdTTmodes = self.nh,
            ttVar = mytt,
            viz = viz,
            ShowViz = True,
            b_print_rate = 1)

        self.train_network(self.mm_ottrnn, self.mm_data, 
                           self.mm_batch_size, self.mm_epochs)


        # tf.reset_default_graph()
        # self.mm_lstm=TFRNN(
        #     name="mm_lstm",
        #     num_in = self.vec_size,
        #     num_hidden = self.hidden_size,
        #     num_out = self.vec_size,
        #     num_target = self.vec_size,
        #     single_output = True,
        #     rnn_cell=tf.contrib.rnn.LSTMCell,
        #     activation_hidden=tf.tanh,
        #     activation_out=tf.identity,
        #     optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
        #     loss_function=tf.squared_difference,
        #     ttRank = self.ttRank,
        #     seq_len = self.seqlen,
        #     imTTmodes = None,
        #     hdTTmodes = self.nh,
        #     viz = self.MyViz)
        # self.train_network(self.mm_lstm, self.mm_data, 
        #                    self.mm_batch_size, self.mm_epochs)

        print('Init and training OTTRNN done.')


    def train_networks(self):
        print('Starting training...')

        # MOVING MNIST
        self.train_ottrnn()
        
        
        print('Done and done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nsamps', type=int, default = 20000)
    parser.add_argument('-e', '--epochs', type=int, default = 100)
    parser.add_argument('-l', '--seqlen', type=int, default = 3)
    parser.add_argument('-b', '--batch_size', type=int, default = 4)
    parser.add_argument('-f', '--frame_size', type=int, default = 64)
    parser.add_argument('-t', '--ttRank', type=int, default = 64)
    parser.add_argument('-d', '--digit_size', type=int, default = 28)
    parser.add_argument('-v', '--speed', type=int, default = 5)
    parser.add_argument('--sh', type=int, default = 1024)
    parser.add_argument('--nx', default = '[4, 16, 16, 4]')
    parser.add_argument('--nh', default = '[4,  8,  8, 4]')
    args = parser.parse_args()


    mymain=Main()
    mymain.init_data(args)
    mymain.train_networks()