import os
import numpy as np
import pickle
import datetime

from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

import skvideo.io

# Custom Functions -----------------------------------------------------------------------------------------------------
from TTRNN import TT_GRU, TT_LSTM
from OTTRNN import OTT_GRU, OTT_LSTM


# Settings:
model_type = 0 # GRU (0) or LSTM (1)
use_TT = 1
use_OTT = 1 # requires use_TT == 1
num_epochs = 100
batch_size = 4

tt_input_shape = [10, 18, 13, 30]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 8, 8, 8, 1]

alpha = 1e-2
dropoutRate = .25


def get_next_batch(inds, mode='train'):

    N = len(inds)
    X = np.zeros((N, GLOBAL_MAX_LEN, 234*100*3), dtype='int8')
    Y = np.zeros((N, 12), dtype='int8')

    for i in range(N):
        if mode=='train':
            read_in = np.load(data_path + 'scaled_actioncliptrain/' + tr_sample_filenames[inds[i]])
        elif mode == 'test':
            read_in = np.load(data_path + 'scaled_actioncliptest/' + te_sample_filenames[inds[i]])

        this_clip = read_in
        # flatten the dimensions 1, 2 and 3
        this_clip = this_clip.reshape(this_clip.shape[0], -1) # of shape (nb_frames, 240*320*3)
        this_clip = (this_clip - 128).astype('int8')   # this_clip.mean()
        X[i] = pad_sequences([this_clip], maxlen=GLOBAL_MAX_LEN, truncating='post', dtype='int8')[0]
        if mode == 'train':
            Y[i] = tr_labels[inds[i]]
        elif mode == 'test':
            Y[i] = te_labels[inds[i]]
    return [X, Y]


# Load the data --------------------------------------------------------------------------------------------------------
np.random.seed(11111986)



# Had to remove due to anonymity
data_path = '/path/to/data/'
write_out_path = '/path/to/res/'

GLOBAL_MAX_LEN = 1496

classes = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 'HandShake',
           'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']


# filter out labels that are not included because their samples are longer than 50 frames:
tr_sample_filenames = os.listdir(data_path + 'scaled_actioncliptrain/')
tr_sample_filenames.sort()

# tr_sample_ids = np.array( [x[-9:-4] for x in tr_sample_filenames] ).astype('int16')

tr_label_filename = data_path + 'labels_train.txt'
tr_labels = np.loadtxt(tr_label_filename, dtype='int16')
tr_label_ids = tr_labels[:, 0]
tr_labels = tr_labels[:, 1::]


# filter out labels that are not included because their samples are longer than 50 frames:
te_sample_filenames = os.listdir(data_path + 'scaled_actioncliptest/')
te_sample_filenames.sort()

# te_sample_ids = np.array( [x[-9:-4] for x in te_sample_filenames] ).astype('int16')

te_label_filename = data_path + 'labels_test.txt'
te_labels = np.loadtxt(te_label_filename, dtype='int16')
te_label_ids = te_labels[:, 0]
te_labels = te_labels[:, 1::]

assert(len(tr_sample_filenames)==len(tr_label_ids))
assert(len(te_sample_filenames)==len(te_label_ids))
num_train_samples = len(tr_sample_filenames)
num_test_samples = len(te_sample_filenames)


# Define the model -----------------------------------------------------------------------------------------------------

input = Input(shape=(GLOBAL_MAX_LEN, 234*100*3))

if model_type == 0:
    if use_TT ==0:
        rnn_layer = GRU(np.prod(tt_output_shape),
                        return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        if use_OTT==0:
            rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                               tt_ranks=tt_ranks,
                               return_sequences=False,
                               dropout=0.25, recurrent_dropout=0.25, activation='tanh')
        else:
            rnn_layer = OTT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                               tt_ranks=tt_ranks,
                               return_sequences=False,
                               dropout=0.25, recurrent_dropout=0.25, activation='tanh')
else:
    if use_TT ==0:
        rnn_layer = LSTM(np.prod(tt_output_shape),
                         return_sequences=False,
                         dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        if use_OTT==0:
            rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                                tt_ranks=tt_ranks,
                                return_sequences=False,
                                dropout=0.25, recurrent_dropout=0.25, activation='tanh')
        else:
            rnn_layer = OTT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                            tt_ranks=tt_ranks,
                            return_sequences=False,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh')
h = rnn_layer(input)
output = Dense(units=12, activation='sigmoid', kernel_regularizer=l2(alpha))(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')


# Start training -------------------------------------------------------------------------------------------------------

num_train_batches = num_train_samples // batch_size
num_test_batches = num_train_samples // batch_size

train_idxs = range(num_train_samples)
test_idxs = range(num_test_samples)

for e in range(num_epochs):

    print 'Starting Epoch ' + str(e)

    # random shuffle for next epoch
    np.random.shuffle(train_idxs)

    # Training batch loop
    train_res = 0
    for b in range(num_train_batches):
        print 'Batch ' + str(b) + '/' + str(num_train_batches)
        batch_idxs = train_idxs[b*batch_size:((b+1)*batch_size)]
        X_batch, Y_batch = get_next_batch(batch_idxs, mode='train')

        model.fit(X_batch, Y_batch, nb_epoch=1, batch_size=batch_size, verbose=1)

        Y_hat = model.predict(X_batch)
        tmp = average_precision_score(Y_batch, Y_hat, average='samples')*len(batch_idxs)
        print str(tmp/len(batch_idxs))
        train_res += tmp
        
    print 'Training Res: ' + str(float(train_res)/num_train_samples)

    if e % 10 == 0:            
        # Testing batch loop
        test_res = 0
        for b in range(num_test_batches):
            batch_idxs = test_idxs[b*batch_size:((b+1)*batch_size)]
            X_batch, Y_batch = get_next_batch(batch_idxs, mode='test')

            Y_hat = model.predict(X_batch)
            test_res += average_precision_score(Y_batch, Y_hat, average='samples')*len(batch_idxs)
            
        print 'Testing Res: ' + str(float(test_res)/num_test_samples)


