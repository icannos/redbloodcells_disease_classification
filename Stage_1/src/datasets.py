import random
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def random_downsampling(l, k):
    '''
    Take a random subset of k elements in a list of length l
    '''
    if l < k:
        return np.asarray(list(range(l)))

    return np.asarray(sorted(random.sample(list(range(l)), k=k)))


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    '''
    Utils to create a window for SSIM
    '''
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    '''
    Utils to compute SSIM
    '''
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    '''
    Computes the SSIM index between two images.
    '''
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def getmax_sim(sim):
    """
    Get the pair of images with the highest similarity. It is used to remove the most similar consecutive images
    """
    m = 0
    idx = None

    for k, x in sim.items():
        if m < x:
            m = x
            idx = k

    return k


# Downsampling function
def ssim_downsampling(sequence, k):
    """
    Perform the downsampling using pytorch (and the GPU, it is really really faster !)
    It computes the ssim between each image and the next one. It then recursively select the most similar pair of images
    and remove one of them. It stops when the number of images is equal to k
    """

    sequence = sequence.to("cuda")
    l = sequence.shape[0]
    keptindices = {i for i in range(l)}

    if l < k:
        return list(keptindices)

    s1 = sequence[:-1]
    s2 = sequence[1:]

    similarities = ssim(s1, s2, size_average=False).squeeze().cpu().detach().numpy()

    similarities = {(i, i + 1): similarities[i] for i in range(0, l - 1)}

    while len(keptindices) > k:
        key = getmax_sim(similarities)
        similarities.pop(key)

        nsim = float(
            ssim(
                sequence[key[0] - 1].unsqueeze(dim=0),
                sequence[key[1]].unsqueeze(dim=0),
                size_average=False,
            )
            .squeeze()
            .cpu()
            .detach()
            .numpy()
        )
        similarities[(key[0] - 1, key[1])] = nsim

        keptindices.remove(key[0])

    return np.asarray(list(keptindices))


def fmax_length(x):
    l = 0

    for s in x:
        ltmp = len(s)
        if ltmp > l:
            l = ltmp
    return l


def pad_collate(batch, l=None):
    """
    Pad a sequence of image with black images
    """
    xx, yy = zip(*batch)

    if l is None or l < 0:
        l = fmax_length(xx)

    X = np.zeros((l, len(batch), 31, 31))

    for i, x in enumerate(xx):
        X[: len(x), i] = x

    yy = np.array(yy)

    return torch.Tensor(X), torch.Tensor(yy).long()


class GlobulesDataset(Dataset):
    """
    Load the data from file and provide an interface for the pytorch code.
    It is the main interface to the data
    """

    def __init__(
        self,
        path,
        task="cleaning",
        maxlength=20,
        preprocessing_method=None,
        unreliable_downsampling=5000,
    ):
        self.path = Path(path)
        self.preprocessing_method = preprocessing_method
        self.maxlength = maxlength
        self.task = task
        self.unreliable_downsampling = unreliable_downsampling

        df = pd.read_csv(self.path / "dataset.csv")
        self.y = df["label"].tolist()
        self.sizes = df["size"].tolist()
        self.names = df["sequence_name"].tolist()

        self.len = len(self.y)

        # Base dataset has 3 labels: 0 "healthy" , 1 "sick/strange", 2 "unreliable"
        if self.task == "cleaning":
            # If cleaning we only have 2 labels: 1 if "unreliable", 0 if reliable
            self.y = np.array((np.array(self.y) == 2), dtype=int)
            if self.unreliable_downsampling:
                self.indices = list(np.argwhere(self.y != 1)[:, 0]) + random.sample(
                    list(np.argwhere(self.y == 1)[:, 0]), k=unreliable_downsampling
                )
                self.len = len(self.indices)
                print(self.len)

        elif self.task == "classification":
            # If classification we remove (true) unreliables
            # Thus we keep only elements which label is 0 or 1 (not 2)
            self.indices = [i for i, v in enumerate(self.y) if v != 2]
            self.len = len(self.indices)
            self.y = np.array(np.array(self.y) == 0, dtype=int)
        else:
            raise Exception("Bad argument")
        # Build a list of list, for each sequence gives a list of the element
        # we want to keep
        if preprocessing_method == "uniform":
            self.kept_elements = self.uniform_downsampling()
        elif preprocessing_method == "similarity":
            self.kept_elements = self.similarity_downsampling()
        else:
            pass

    def class_counts(self):
        if self.task == "classification":
            unique, counts = np.unique(self.y[self.indices], return_counts=True)
        else:
            if self.task == "cleaning" and self.unreliable_downsampling:
                unique, counts = np.unique(self.y[self.indices], return_counts=True)
            else:
                unique, counts = np.unique(self.y, return_counts=True)

        return counts

    def uniform_downsampling(self):
        keptelements = []
        for item in tqdm(range(self.len)):
            if self.task == "classification":
                item = self.indices[item]
            if self.task == "cleaning" and self.unreliable_downsampling:
                item = self.indices[item]

            if self.sizes[item] > self.maxlength:
                keptelements.append(
                    random_downsampling(self.sizes[item], self.maxlength)
                )
            else:
                keptelements.append([i for i in range(self.sizes[item])])

        return keptelements

    def similarity_downsampling(self):
        keptelements = []
        for item in tqdm(range(self.len)):
            if self.task == "classification":
                item = self.indices[item]
            if self.task == "cleaning" and self.unreliable_downsampling:
                item = self.indices[item]

            if self.sizes[item] > self.maxlength:
                im = np.array(
                    Image.open(self.path / "img" / (self.names[item] + ".png"))
                )
                im = torch.Tensor(im).view(31, 31, self.sizes[item]).permute(2, 0, 1)
                keptelements.append(
                    ssim_downsampling(im.unsqueeze(dim=1), self.maxlength)
                )
            else:
                keptelements.append([i for i in range(self.sizes[item])])

        return keptelements

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        idx = item
        if self.task == "classification":
            idx = self.indices[item]
        if self.task == "cleaning" and self.unreliable_downsampling:
            idx = self.indices[item]

        im = np.array(Image.open(self.path / "img" / (self.names[idx] + ".png")))

        if self.preprocessing_method is not None:
            im = np.asarray(
                [im[:, i * 31 : (i + 1) * 31] for i in self.kept_elements[item]]
            )

        else:
            im = np.asarray(
                [im[:, i * 31 : (i + 1) * 31] for i in range(self.sizes[idx])]
            )

        return im / 255, self.y[idx]

    def get_class(self, item):
        idx = item
        if self.task == "classification":
            idx = self.indices[item]
        if self.task == "cleaning" and self.unreliable_downsampling:
            idx = self.indices[item]

        return self.y[idx]

    def get_seq(self, item, preprocessed=False):
        idx = item
        if self.task == "classification":
            idx = self.indices[item]
        if self.task == "cleaning" and self.unreliable_downsampling:
            idx = self.indices[item]

        im = np.array(Image.open(self.path / "img" / (self.names[idx] + ".png")))

        if self.preprocessing_method is not None:
            im = np.asarray(
                [im[:, i * 31 : (i + 1) * 31] for i in self.kept_elements[item]]
            )
            l = int(min(self.sizes[idx], self.maxlength))

        else:
            im = np.asarray(
                [im[:, i * 31 : (i + 1) * 31] for i in range(self.sizes[idx])]
            )
            l = min(self.sizes[idx])

        nim = np.zeros((31, 31 * l))
        for i in range(l):
            nim[:, i * 31 : (i + 1) * 31] = im[i]

        return nim / 255, self.y[idx]
