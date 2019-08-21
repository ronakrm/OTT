# process hollywood2 labels into 12-dim vector of binary classifiers

# generate a labels_train.txt and a labels_test.txt

import numpy as np
import csv
import os

datapath = '/home/ronak/aott/TT_RNN/Datasets/Hollywood2/'
labelpath = datapath+'ClipSets/'

classes = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 'HandShake',
           'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']


for split in ['train', 'test']:

    # get first column of filenames
    fnames = os.listdir(datapath + 'actionclip' + split + '/')
    fnames.sort()

    ids = np.zeros([len(fnames), 1], dtype='int16')
    for i,x in zip(range(0,len(fnames)), fnames):
        ids[i,0] = int(x[-9:-4])

    labels = np.zeros([len(fnames), 12], dtype='int16')

    # get class labels
    for j,myclass in zip(range(0,len(classes)),classes):
        myfile = open(labelpath + myclass + '_' + split + '.txt','r')
        reader = csv.reader(myfile, delimiter=' ')
        for i,row in zip(range(0,len(fnames)), reader):
            if row[2] == str(-1):
                labels[i,j] = 0
            elif row[2] == str(1):
                labels[i,j] = 1
            else:
                print('unknown label')
                print(row[2])
                exit(0)



    data = np.concatenate((ids, labels), axis=1)

    np.savetxt(datapath+'labels_'+split+'.txt', data, fmt='%d')