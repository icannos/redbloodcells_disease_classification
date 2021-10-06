'''
This file defines the dataset class, they supposed raw data have been preprocessed into picklable files
containing a list of sequences and a list of label corresponding to the sequences.
'''
from torch.utils.data import Dataset
import pickle as pk
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

class GlobulesDatasetCleaning(Dataset):
    '''
    Load the data from file and provide an interface for the pytorch code.

    '''
    def __init__(self, path, n_classes=3):
        with open(path, "rb") as f:
            self.x, self.y, self.names = pk.load(f, encoding="bytes")

        if n_classes == 3:
            pass
        else:
            self.y = np.array((np.array(self.y) == 2), dtype=int)
            print(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item] / 255, self.y[item]

class GlobulesDatasetUsable(Dataset):
    '''
    Load the data from file and provide an interface for the pytorch code.
    '''
    def __init__(self, path, stage_one_output=None):
        with open(path, "rb") as f:
            self.x, self.y, self.names = pk.load(f, encoding="bytes")

            if stage_one_output == None:
                x = [self.x[i] for i in range(len(self.x)) if self.y[i] != 2]
                y = [self.y[i] for i in range(len(self.x)) if self.y[i] != 2]
                names = [self.names for i in range(len(self.x)) if self.y[i] != 2]
            else:
                x = [self.x[i] for i in range(len(self.x)) if stage_one_output[i] == 0]
                y = [self.y[i] for i in range(len(self.x)) if stage_one_output[i] == 0]
                names = [self.names for i in range(len(self.x)) if stage_one_output[i] == 0]

            self.x = x
            self.y = np.asarray(y, dtype=int)
            self.names = names

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item] / 255, self.y[item]

def max_length(x):
    l = 0

    for s in x:
        ltmp = len(s)
        if ltmp > l:
            l = ltmp
    return l

def pad_collate(batch, l=None):
    '''
    Pad a sequence of image with black images
    :param batch:
    :return:
    '''
    xx, yy = zip(*batch)

    if l is None or l < 0:
        l = max_length(xx)

    X = np.zeros((l, len(batch), 31,31))

    for i, x in enumerate(xx):
        X[:len(x), i] = x

    yy = np.array(yy)

    return X, torch.Tensor(yy).long()