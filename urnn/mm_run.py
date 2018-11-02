import argparse
import numpy as np
import tensorflow as tf
from problems.movingmnist import MovingMnistProblemDataset
from networks.tf_rnn import TFRNN
from networks.ottrnn_cell import OTTRNNCell

glob_learning_rate = 0.001*10
glob_decay = 0.9

class Main:
    def init_data(self, args):
        print('Generating data...')

        self.mm_batch_size = args.batch_size
        self.mm_epochs = args.epochs
        self.nsamps = args.nsamps
        self.seqlen = args.seqlen

        self.hidden_size = args.size
        self.nh = args.nh
        self.frame_size = args.size
        self.nx = args.nx
        self.ttRank = args.ttRank
        self.digit_size = args.digit_size
        self.speed = args.speed


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
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=',sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
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
            loss_function = tf.squared_difference,
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
        self.train_ottrnn()
        
        
        print('Done and done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nsamps', type=int, default = 20000)
    parser.add_argument('-e', '--epochs', type=int, default = 100)
    parser.add_argument('-l', '--seqlen', type=int, default = 3)
    parser.add_argument('-b', '--batch_size', type=int, default = 4)
    parser.add_argument('-s', '--size', type=int, default = 64)
    parser.add_argument('-t', '--ttRank', type=int, default = 64)
    parser.add_argument('-d', '--digit_size', type=int, default = 28)
    parser.add_argument('-v', '--speed', type=int, default = 5)
    parser.add_argument('--nx', default = [4, 16, 16, 4])
    parser.add_argument('--nh', default = [2,  4,  4, 2])
    args = parser.parse_args()


    mymain=Main()
    mymain.init_data(args)
    mymain.train_networks()