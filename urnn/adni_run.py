import argparse
import numpy as np
import ast
import tensorflow as tf
from problems.adni import ADNIProblemDataset
from networks.tf_rnn import TFRNN
from networks.ottrnn_cell import OTTRNNCell

glob_learning_rate = 0.001
glob_decay = 0.9
np.random.seed(0)

### DIMENSION OF 3D VOLUME
dX = 121
dY = 145
dZ = 121

class Main:
    def init_data(self, args):
        print('Initializing data...')

        self.mm_batch_size = args.batch_size
        self.mm_epochs = args.epochs
        self.seqlen = 3
        self.hidden_size = args.sh
        self.nh = ast.literal_eval(args.nh)
        self.nx = ast.literal_eval(args.nx)
        self.ttRank = int(args.ttRank)

        self.datapath = '/media/ronak/SSD_1T_ADNI/ADNI/MRI3_Seqs/'

        self.vec_size = dX*dY*dZ

        assert(np.prod(self.nh)==self.hidden_size)

        self.mm_data= ADNIProblemDataset(datapath=self.datapath)

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
    parser.add_argument('-e', '--epochs', type=int, default = 100)
    parser.add_argument('-b', '--batch_size', type=int, default = 4)
    parser.add_argument('-t', '--ttRank', type=int, default = 64)
    parser.add_argument('--sh', type=int, default = 1024)
    parser.add_argument('--nx', default = '[4, 16, 16, 4]')
    parser.add_argument('--nh', default = '[4,  8,  8, 4]')
    args = parser.parse_args()


    mymain=Main()
    mymain.init_data(args)
    mymain.train_networks()