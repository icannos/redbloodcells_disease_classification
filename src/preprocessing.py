from scipy.io import loadmat
from collections import defaultdict
import numpy as np
import os
import pickle as pk
import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from multiprocessing import Pool
from pathlib import Path
import random


"""
Code of Po-Hsun-Su
https://github.com/Po-Hsun-Su/pytorch-ssim

Implementation of the Structural similarity in pytorch
"""

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    '''
    Computes the ssim metric using torch and cuda
    '''
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

"""
Our code for preprocess our data
"""


def parallel_downsampling(sequence, k):
    '''
    Perform the downsampling using pytorch (and the GPU, it is really really faster !)
    :param sequence:
    :param k:
    :return:
    '''

    sequence = sequence.to('cuda')
    while sequence.shape[0] > k:
        s1 = sequence[:-1]
        s2 = sequence[1:]

        similarities = ssim(s1, s2, size_average=False)

        x = torch.argmax(similarities)

        sequence = torch.cat([sequence[:x], sequence[x+1:]])

    return sequence.squeeze().cpu().detach().numpy()

#  [patient name][date]_[video num]_[sequence num].jpg -> e.g. BK0926_10s01_172.jpg
def raw_sequences(path, cut=True):
    '''
    Read the data from mat file and remove the black images used to pad
    :param path:
    :return: X = [(seq_len, 31,31), ..] list of sequences, Y = [labels]
    '''
    mat = loadmat(path)

    seqs = mat['Norm_Tab']
    labels = mat['Labels_Num']
    path = Path(path)

    # BG20191003shear10s02_Export.mat

    img_basename = path.stem[0:2] + path.stem[6:10] + "_" + path.stem[15:20] + "_"

    # labels[i][0]
    # seqs[i, j] = image j of seq i

    X = []
    Y = []
    names = []

    for i in range(seqs.shape[0]):
        s = []
        for j in range(seqs[i].shape[0]):
            if cut and j > 128:
                break
            if np.all(seqs[i, j] == np.zeros((31,31))):
                break
            s.append(seqs[i, j])

        if s is not None:
            X.append(np.array(s))
            Y.append(np.array(labels[i][0]))
            names.append(img_basename + str(i))

    del(mat)

    return X, Y, names

def downsampled_sequences(path, k):
    '''
    Downsample the sequences to k elements
    :param path:
    :param k:
    :return:
    '''
    print("Loading...")
    X, Y, names = raw_sequences(path, cut=False)
    print("Begin downsampling")

    X2 = []
    Y2 = []
    names2 = []

    for i in range(len(X)):
        sequence = torch.unsqueeze(torch.Tensor(X[i]).float(), 1)

        s = parallel_downsampling(sequence, k)

        if s is not None:
            X2.append(np.array(s))
            Y2.append(np.array(Y[i]))
            names2.append(names[i])

    return X2, Y2, names

def uniformly_downsampled(path, k):
    '''
    Downsample the sequences to k elements
    :param path:
    :param k:
    :return:
    '''
    print("Loading...")
    X, Y , names = raw_sequences(path, cut=False)
    print("Begin downsampling")

    X2 = []
    Y2 = []
    names2 = []

    for i in range(len(X)):
        seqlen = len(X[i])
        if seqlen > k:
            idx = random.sample(list(range(seqlen)), k)

            s = X[i][idx]
        else:
            s = X[i]

        if s is not None:
            X2.append(np.array(s))
            Y2.append(np.array(Y[i]))
            names2.append(names[i])

    return X2, Y2, names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts the globule sequences from the mat files.')
    parser.add_argument('path', type=str, help='Path to the directory or file to process.')
    parser.add_argument('output', type=str, help='Output path, directory where all the files will be put if the input'
                                                 'is a directory or file if the input was a file.')

    parser.add_argument('-m', '--merge', default=False, action="store_true",
                        help='If a directory is specified as input, it will merge all the data in one single file.'
                             'Warning: It could use a lot of RAM.')

    parser.add_argument('-t', '--test', default=False, action="store_true",
                        help='Keep original data alongside the downsampled data for comparison later on.')

    parser.add_argument('-s', '--similarity', default=False, type=int,
                        help='If set the sequence is downsampled to k elements removing images that are too close')
    parser.add_argument('-u', '--uniformly', default=False, type=int,
                        help='If set the sequence is downsampled to k elements removing images randomly')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
    else:
        files = [args.path] if os.path.isfile(args.path) else []

    if not args.merge and len(files) > 0:
        for i, f in enumerate(files):
            print(f)
            if args.similarity:
                d = downsampled_sequences(f, args.similarity)
            elif args.uniformly:
                d = uniformly_downsampled(f, args.uniformly)
            else:
                d = raw_sequences(f)

            pk.dump(d, open(args.output.format(i=i), "wb"), protocol=pk.HIGHEST_PROTOCOL)

    elif args.merge:
        X = []
        Y = []
        names=[]

        for i, f in enumerate(files):
            print(f)
            if args.similarity:
                X_tmp, Y_tmp, names_tmp = downsampled_sequences(f, args.similarity)
            elif args.uniformly:
                X_tmp, Y_tmp, names_tmp = uniformly_downsampled(f, args.uniformly)
            else:
                X_tmp, Y_tmp, names_tmp = raw_sequences(f)


            X += X_tmp
            Y += Y_tmp
            names += names_tmp

        pk.dump((X,np.array(Y), names), open(args.output, "wb"), protocol=pk.HIGHEST_PROTOCOL)


