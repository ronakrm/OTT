# rescale hollywood2 videos to 234 wide by 100 tall using ffmpeg,
# save as avi in scaled folder

import os
import skvideo.io
import subprocess as sp
import numpy as np
# import pickle

datapath = '/media/ronak/SSD_1T_ADNI/Hollywood2/'

newheight = 100
newwidth = 234

for split in ['test']:#, 'train']:

    # folder of videos
    infolder = datapath + 'actionclip' + split + '/'

    # output folder
    outfolder = datapath + 'scaled_actionclip' + split + '/'

    # list of videos
    fnames = os.listdir(infolder)
    fnames.sort()

    # for all videos
    for sample in fnames:
        print('Processing Sample %d', sample)

        # inpath = infolder+sample
        outpath = outfolder+sample

        # formatspec = "/usr/bin/ffmpeg -y -i " + inpath + " -vf \"scale="+str(newwidth)+":"+str(newheight)+"\" " + outpath

        # print(formatspec)

        # resize the video with ffmpeg
        # sp.call(['ffmpeg','-y','-i',inpath,'-vf','scale='+str(newwidth)+':'+str(newheight),outpath])


        ### if you want to save to npy
        ### WARNING: file sizes are large, better to load videos per batch into
        ###          array as needed

        # # load the video to a numpy array
        videodata = skvideo.io.vread(outpath)
        np.save(outpath, videodata, allow_pickle=True)

