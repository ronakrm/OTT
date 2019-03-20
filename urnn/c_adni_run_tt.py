import argparse
import numpy as np
import ast
import tensorflow as tf
from problems.c_adni import ADNIProblemDataset
from networks.adni_tfrnn import TFRNN
from networks.ttrnn_cell import TTRNNCell
import time
glob_learning_rate = 1e-5
glob_decay = 0.9
np.random.seed(0)
tf.random.set_random_seed(0)

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

        self.datapath = '/home/ronak/mntpt4/ronak/ADNI/r_MRI3_Seqs/'

        self.vec_size = dX*dY*dZ

        # assert(np.prod(self.nh)==self.hidden_size)

        self.mm_data= ADNIProblemDataset(datapath=self.datapath)

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        sample_len = str(dataset.get_sample_len())
        print('Training network ', net.name, '... timesteps=',sample_len)
        net.train(dataset, batch_size, epochs)
        # loss_list has one number for each batch (step)
        print('Training network ', net.name, ' done.')


    def train_ttrnn(self):
        print('Initializing and training TTRNN...')

        tf.reset_default_graph()
        self.c_adni_ttrnn = TFRNN(
            name = "c_adni_ttrnn",
            num_in = self.vec_size,
            num_hidden = self.hidden_size,
            num_out = 1,
            num_target = 1,
            single_output = False,
            rnn_cell = TTRNNCell,
            activation_hidden = tf.tanh, # modReLU
            activation_out = tf.identity,
            optimizer = tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
            loss_function = tf.squared_difference,
            seq_len = self.seqlen,
            ttRank = self.ttRank,
            imTTmodes = self.nx,
            hdTTmodes = self.nh)
        self.train_network(self.c_adni_ttrnn, self.mm_data, 
                           self.mm_batch_size, self.mm_epochs)

        print('Init and training TTRNN done.')


    def train_networks(self):
        print('Starting training...')

        t0 = time.time()
        self.train_ttrnn()
        t1 = time.time()
        myT = t1 - t0
        
        print('Done and done.')
        print('Full Training Took',myT,'seconds.')

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
