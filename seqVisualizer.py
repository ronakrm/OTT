import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class seqVisualizer():
    def __init__(self, batch_size, seqlen, frame_size):
        # Initialize subplots so we can use set_data for repeated plotting inside the loop
        nViz = np.min([int(batch_size), 12])
        self.nsrows = int(nViz/4)# 3
        self.nscols = int(nViz/3)# 4

        self.num_samples = self.nsrows * self.nscols
        self.nrows = self.nsrows * 2 # 2 here is gt and pred
        self.seqlen = seqlen
        self.ncols = self.nscols * self.seqlen
        self.frame_size = frame_size
        self.fig, self.axarr = plt.subplots(self.nrows, self.ncols, figsize=(23,15)) 
        fig_arr = [[mpimg.AxesImage for j in range(self.ncols)] for i in range(self.nrows)]
        for ii in range(self.nrows):
            for jj in range(self.ncols):
                rand_mat = np.random.rand(self.frame_size, self.frame_size)
                self.axarr[ii, jj].axis('off')
                fig_arr[ii][jj] = self.axarr[ii, jj].imshow(rand_mat, cmap='gray')

    def updateViz(self, gt, pd, showSome=True):
        # visualize the first num_sample of the test set
        s = 0
        for r in range(0,self.nrows):
            for c in range(0,self.ncols):
                c_idx = int(np.floor(c / self.seqlen))
                r_idx = int(np.floor(r / 2))
                sample = r_idx*self.nscols + c_idx

                frame = s % self.seqlen

                # PRED
                if r % 2 == 1:
                    tmp = np.flipud(np.rot90(np.reshape(pd[sample,frame,:], [self.frame_size, self.frame_size])))
                # GT
                else:
                    tmp = np.flipud(np.rot90(np.reshape(gt[sample,frame,:], [self.frame_size, self.frame_size])))
                self.axarr[r, c].clear()
                self.axarr[r, c].imshow(tmp, clim=(0.0, 1.0))
                self.axarr[r, c].axis('off')

                s = s + 1

        # this shows the figure
        if showSome:
            plt.pause(0.1)