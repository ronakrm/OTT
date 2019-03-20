import argparse
import numpy as np
import ast
import tensorflow as tf
# from problems.s_movingmnist import MovingMnistProblemDataset
from problems.movingmnist import MovingMnistProblemDataset
from networks.tfrnn import TFRNN

from networks.ttrnn_straight import TTRNNCell as mycell
# from networks.rnn_straight import myRNNCell as mycell
from vars.sOTTtfVariable import sOTTtfVariable as myott
from vars.TTtfVariable import TTtfVariable as mytt

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

        if args.ott1 == True:
            self.tVar1 = myott
        else:
            self.tVar1 = mytt
        if args.ott2 == True:
            self.tVar2 = myott
        else:
            self.tVar2 = mytt
        

        self.name = "mm_ottrnn_" + str(self.seqlen) + "_" + str(args.ott1) + "_" + str(args.ott2)

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
            name = self.name,
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
            ttVar1 = self.tVar1,
            ttVar2 = self.tVar2,
            viz = viz,
            ShowViz = False,
            b_print_rate = 1)

        self.train_network(self.mm_ottrnn, self.mm_data, 
                           self.mm_batch_size, self.mm_epochs)

        print('Init and training OTTRNN done.')


    def train_networks(self):
        print('Starting training...')

        # MOVING MNIST
        self.train_ottrnn()
        
        
        print('Done and done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nsamps', type=int, default = 1000)
    parser.add_argument('-e', '--epochs', type=int, default = 100)
    parser.add_argument('-l', '--seqlen', type=int, default = 3)
    parser.add_argument('-b', '--batch_size', type=int, default = 16)
    parser.add_argument('-f', '--frame_size', type=int, default = 256)
    parser.add_argument('-d', '--digit_size', type=int, default = 112)
    parser.add_argument('-t', '--ttRank', type=int, default = 16)
    parser.add_argument('-v', '--speed', type=int, default = 25)
    parser.add_argument('--sh', type=int, default = 4096)
    parser.add_argument('--nx', default = '[16, 16, 16, 16]')
    parser.add_argument('--nh', default = '[4,  16,  16, 4]')
    parser.add_argument('--ott1', action='store_true', default=False)
    parser.add_argument('--ott2', action='store_true', default=False)
    args = parser.parse_args()


    mymain=Main()
    mymain.init_data(args)
    mymain.train_networks()