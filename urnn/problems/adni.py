import numpy as np
from .dataset import Dataset
from PIL import Image
from pathlib import Path
import sys
import os
import math

### DIMENSION OF 3D VOLUME
dX = 121
dY = 145
dZ = 121

# vectorized is return, check reshape lines for changing this
class ADNIProblemDataset(Dataset):

    def __init__(self, datapath):

        self.datapath = datapath
        if Path(self.datapath).exists():
            pass
        else:
            error('ADNI path specified does not exist.')

        self.vectdim = int(dX*dY*dZ) # vectorized size

        # each folder in ADNI folder should be a separate subject
        self.tot_samples = len(os.listdir(self.datapath))

        np.random.permutation(self.tot_samples)
        self.idxs = np.random.permutation(self.tot_samples)
        self.train_size = 400
        self.num_samples = self.train_size
        self.train_idxs = self.idxs[:self.train_size]
        self.valid_size = 100
        self.valid_idxs = self.idxs[self.train_size:self.train_size+self.valid_size]
        self.test_size = self.tot_samples - (self.train_size + self.valid_size)
        self.test_idxs = self.idxs[self.train_size+self.valid_size:]

        # 3 timepoints HARDCODED HERE
        self.sample_len = 3
        
        if self.tot_samples < 1:
            print('found no files in specified ADNI path.')
            sys.exit(0)

    def get_batch_count(self, batch_size):
        return self.num_samples // batch_size

    def get_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        fnames = [self.datapath+str(idx)+'.npy' for idx in self.train_idxs[start_idx:end_idx]]
        X_batch = self.load_data(fnames)

        return X_batch, X_batch

    def get_test_data(self):
        fnames = [self.datapath+str(idx)+'.npy' for idx in self.test_idxs]
        X_batch = self.load_data(fnames)
        
        return X_batch, X_batch

    def get_validation_data(self, nsamps):
        fnames = [self.datapath+str(idx)+'.npy' for idx in self.valid_idxs[:nsamps]]
        X_batch = self.load_data(fnames)

        return X_batch, X_batch

    def load_data(self, fnames):
        dat = np.empty((len(fnames), self.sample_len, dX, dY, dZ), dtype=np.float32)
        for f,i in zip(fnames, range(0,len(fnames))):
            tmp = np.load(f)
            dat[i,:,:,:,:] = tmp
        X = np.reshape(dat, [-1, self.sample_len, self.vectdim])
        return X