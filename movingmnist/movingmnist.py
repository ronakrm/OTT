import numpy as np
from dataset import Dataset
from PIL import Image
from pathlib import Path
import sys
import os
import math

# vectorized is return, check reshape lines for changing this
class MovingMnistProblemDataset(Dataset):

    def __init__(self, num_samples, sample_len, datapath, num_sz, frame_size, speed):

        self.datapath = datapath
        if Path(self.datapath).exists():
            pass
        else:
            os.makedirs(self.datapath)

        self.mnist = self.load_dataset()
        self.num_sz = num_sz # MNIST digit size
        self.ns_p_img = 1 # number of MNIST images per frame
        self.frame_size = frame_size # frame size
        self.frame_shape = (self.frame_size, self.frame_size) # frame size
        self.vectdim = int(self.frame_size**2) # vectorized size
        self.speed = speed # moving rate by pixel

        self.num_samples = num_samples
        self.sample_len = sample_len
        if num_samples == -1:
            return
        self.generate_training_data(int(self.num_samples))

    def get_batch_count(self, batch_size):
        return self.num_samples // batch_size

    def get_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        fnames = [self.datapath+str(i)+'.npy' for i in range(start_idx, end_idx)]
        X_batch = self.load_training_data(fnames)

        return X_batch, X_batch

    def get_test_data(self):
        self.X_test, self.Y_test = self.generate(int(num_samples * 0.2))
        return self.X_test, self.Y_test

    def get_validation_data(self, nsamps):
        X, Y = self.generate(int(nsamps))
        return X, Y

    def generate(self, num_samples):
        width, height = self.frame_shape
        dat = np.empty((num_samples, self.sample_len, width, height), dtype=np.float32)
        for idx in range(num_samples):
            samp = self.generate_moving_mnist_sample()
            dat[idx,:,:,:] = np.squeeze(samp)

        X = np.reshape(dat, [-1, self.sample_len, self.vectdim])
        return X, X


    def load_training_data(self, fnames):
        width, height = self.frame_shape
        dat = np.empty((len(fnames), self.sample_len, width, height), dtype=np.float32)
        for f,i in zip(fnames, range(0,len(fnames))):
            tmp = np.load(f)
            dat[i,:,:,:] = tmp
        X = np.reshape(dat, [-1, self.sample_len, self.vectdim])
        return X

    def generate_training_data(self, seqs=1):
        # width, height = self.frame_shape
        # dataset = np.empty((seqs, self.sample_len, width, height), dtype=np.float32)
        for seq_idx in range(seqs):
            fname = self.datapath+str(seq_idx)+'.npy'
            if Path(fname).exists():
                continue
            else:
                samp = self.generate_moving_mnist_sample()
                np.save(file=fname, arr=samp)
            #dataset[seq_idx,:,:,:] = np.squeeze(samp)



    ## following mostly taken from Tencia Lee via GitHub, from
    # [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
    #     Srivastava et al
    # helper functions

    # generates and returns video frames in uint8 array
    def generate_moving_mnist_sample(self):
        mnist = self.mnist
        width, height = self.frame_shape
        lims = (x_lim, y_lim) = width-self.num_sz, height-self.num_sz

        sample = np.empty((self.sample_len, width, height), dtype=np.float32)
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(self.ns_p_img)*2 - 1)
        # speeds = np.random.randint(5, size=self.ns_p_img)+2
        speeds = np.array(self.speed*np.ones(self.ns_p_img))
        veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in zip(direcs, speeds)]
        mnist_images = [Image.fromarray(self.get_picture_array(mnist,r,shift=0)).resize((self.num_sz,self.num_sz), Image.ANTIALIAS) \
               for r in np.random.randint(0, mnist.shape[0], self.ns_p_img)]
        positions = [(np.random.rand()*x_lim, np.random.rand()*y_lim) for _ in range(self.ns_p_img)]
        for frame_idx in range(self.sample_len):
            canvases = [Image.new('L', (width,height)) for _ in range(self.ns_p_img)]
            canvas = np.zeros((1,width,height), dtype=np.float32)
            for i,canv in enumerate(canvases):
                coooords = tuple([int(round(p)) for p in positions[i]])#map(lambda p: int(round(p)), positions[i]))
                canv.paste(mnist_images[i], coooords)
                canvas += self.arr_from_img(canv, shift=0)
            # update positions based on velocity
            next_pos = [(p[0]+v[0],p[1]+v[1]) for p,v in zip(positions, veloc)]
            # bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j]+2:
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))
            positions = [(p[0]+v[0],p[1]+v[1]) for p,v in zip(positions, veloc)]
            # copy additive canvas to data array
            sample[frame_idx,:,:] = np.squeeze(canvas)
            
        return np.round(sample)

    # loads mnist from web on demand
    def load_dataset(self):
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)
        import gzip
        def load_mnist_images(filename):
            if not os.path.exists(filename):
                download(filename)
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
            return data / np.float32(255)
        return load_mnist_images('data/train-images-idx3-ubyte.gz')

    def arr_from_img(self, im, shift=0):
        w,h=im.size
        arr=im.getdata()
        c = int(np.product(arr.size) / (w*h))
        return np.asarray(arr, dtype=np.float32).reshape((h,w,c)).transpose(2,1,0) / 255. - shift

    def get_picture_array(self, X, index, shift=0):
        ch, w, h = X.shape[1], X.shape[2], X.shape[3]
        ret = ((X[index]+shift)*255.).reshape(ch,w,h).transpose(2,1,0).clip(0,255).astype(np.uint8)
        if ch == 1:
            ret=ret.reshape(h,w)
        return ret
