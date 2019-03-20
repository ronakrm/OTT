import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from seqVisualizer import seqVisualizer

### DIMENSION OF 3D VOLUME
dX = 121
dY = 145
dZ = 121

class ADNIVisualizer(seqVisualizer):
    def __init__(self, batch_size, seqlen, frame_size, yslice):
        super(ADNIVisualizer,self).__init__(batch_size, seqlen, frame_size)

        self.yslice = yslice

        return


    def updateViz(self, gt, pd):
        s = 0
        for r in range(0,self.nrows):
            for c in range(0,self.ncols):
                c_idx = int(np.floor(c / self.seqlen))
                r_idx = int(np.floor(r / 2))
                sample = r_idx*self.nscols + c_idx

                frame = s % self.seqlen

                # PRED
                if r % 2 == 1:
                    tmp = np.reshape(pd[sample,frame,:].T, [dX, dY, dZ])
                    tmp = np.squeeze(tmp[self.yslice, :, :])
                # GT
                else:
                    tmp = np.reshape(gt[sample,frame,:].T, [dX, dY, dZ])
                    tmp = np.squeeze(tmp[self.yslice, :, :])
                self.axarr[r, c].clear()
                self.axarr[r, c].imshow(tmp, clim=(0.0, 1.0))
                self.axarr[r, c].axis('off')

                s = s + 1

        return