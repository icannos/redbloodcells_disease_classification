'''
In this file we store the models architectures we tested.
'''

import torch
import torch.nn as nn
from typing import *


import numpy as np


class RecurrentConvNet(nn.Module):
    """
    This implement an optimized version of the recurrent convolutionnal network in order to ease memory consuption.
    Instead of computing the feature map for each image and then feed all these features into the rnn we compute the feature map
    and then directly pass it to the recurrent network and then we compute the second feature map and so on.
    """

    def __init__(self, n_classes=3, device='cpu', softmax=True):
        '''

        :param n_classes: In our setup we can work on two problem: a classification in 3 classes (healthy, sick, garbage)
        but in some experiments we only try to classify ( (healthy, sick) and garbage) for further processing.
        :param device: cpu or cuda.
        '''
        super().__init__()

        self.softmax = softmax
        self.device = device
        self.hidden = 64

        # The convnet for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1,8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3),
            nn.ReLU(),
        )

        # Size of the flatten tensor from images of size 31x31
        self.rnnCell = nn.GRUCell(1152, self.hidden)

        # Two recurrent network to be able to keep informations in the long run for long sequences
        self.rnn = nn.GRU(self.hidden, self.hidden)

        # The simple classifier at the end of the processus
        self.classifier = nn.Sequential(nn.Linear(self.hidden, 32), nn.Tanh(), nn.Linear(32, n_classes))

    def forward(self, x):
        # We assume x = (seq_length, batch, 31, 31)
        x = torch.unsqueeze(x, dim=2)
        # x = (seq_length, batch, 1, 31, 31)

        batch_size = x.shape[1]
        seq_length = x.shape[0]

        hidden = torch.zeros((batch_size, self.hidden)).to(self.device)

        hiddens = [hidden]

        for i in range(seq_length):
            # Here we compute each feature map of the sequence independently.
            # This ease the memory consumption

            features = self.feature_extractor(x[i].float())
            # features = (batch, 64, h, w)

            features = torch.flatten(features, start_dim=1, end_dim=-1)

            hidden = self.rnnCell(features, hidden)
            hiddens.append(hidden)

        # We run a second layer of recurrent network to be able to retains informations from longer sequences
        output, hidden = self.rnn(torch.stack(hiddens))

        output = self.classifier(output[-1])

        if self.softmax:
            return nn.functional.log_softmax(output, dim=1)
        else:
            return output


class FixedSizeConvnet(nn.Module):
    '''
    Simple convnet model which uses channel to embedded the sequence
    '''
    def __init__(self, in_channels, n_classes, softmax : bool = True, device='cpu'):

        super().__init__()

        self.softmax = softmax
        self.device = device
        self.in_channels = in_channels

        # The convnet for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels,8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(nn.Linear(1152, 32), nn.Tanh(), nn.Linear(32, n_classes))

    def forward(self, x):
        # We assume x = (seq_length, batch, 31, 31)
        x = torch.transpose(x, 0,1)
        # x = batch, seqlength, 31,31
        # seqlength = in_channel in that case

        features = torch.flatten(self.cnn(x), start_dim=1)
        # features = (batch, features_size)

        return self.classifier(features)












