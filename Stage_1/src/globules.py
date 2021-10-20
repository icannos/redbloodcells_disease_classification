"""
Author Maxime Darrin

Code for paper: ...

"""
import argparse

from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from math import exp
from torch.autograd import Variable
from tqdm import tqdm
import datetime
import random

from datasets import fmax_length, pad_collate, GlobulesDataset


class RecurrentConvNet(nn.Module):
    """
    This implement an optimized version of the recurrent convolutionnal network in order to ease memory consuption.
    Instead of computing the feature map for each image and then feed all these features into the rnn we compute the feature map
    and then directly pass it to the recurrent network and then we compute the second feature map and so on.
    """

    def __init__(self, n_classes=3, device="cpu", softmax=True):
        """

        :param n_classes: In our setup we can work on two problem: a classification in 3 classes (healthy, sick, garbage)
        but in some experiments we only try to classify ( (healthy, sick) and garbage) for further processing.
        :param device: cpu or cuda.
        """
        super().__init__()

        self.softmax = softmax
        self.device = device
        self.hidden = 256

        # The convnet for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3),
            nn.ReLU(),
        )

        self.rnnCell = nn.GRUCell(2304, self.hidden)

        # Two recurrent network to be able to keep informations in the long run for long sequences
        self.rnn = nn.GRU(self.hidden, self.hidden)

        # The simple classifier at the end of the processus
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # We assume x = (seq_length, batch, 31, 31)
        x = torch.unsqueeze(x, dim=2)
        # x = (seq_length, batch, 1, 31, 31)

        batch_size = x.shape[1]
        seq_length = x.shape[0]

        hidden = torch.zeros((batch_size, self.hidden)).to(self.device)
        cell = torch.zeros((batch_size, self.hidden)).to(self.device)

        hiddens = [hidden]
        for i in range(seq_length):
            # Here we compute each feature map of the sequence independently.

            features = self.feature_extractor(x[i].float())
            # features = (batch, 64, h, w)

            features = torch.flatten(features, start_dim=1, end_dim=-1)

            hidden = self.rnnCell(features, hidden)
            hiddens.append(hidden)

        output, hidden = self.rnn(torch.stack(hiddens))

        output = self.classifier(output[-1])

        if self.softmax:
            return nn.functional.log_softmax(output, dim=1)
        else:
            return output


class FixedSizeConvnet(nn.Module):
    def __init__(self, in_channels, n_classes, softmax: bool = True, device="cpu"):
        super().__init__()

        self.softmax = softmax
        self.device = device
        self.in_channels = in_channels

        # The convnet for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
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

        self.classifier = nn.Sequential(
            nn.Linear(1152, 32), nn.Tanh(), nn.Linear(32, n_classes)
        )

    def forward(self, x):
        # We assume x = (seq_length, batch, 31, 31)
        x = torch.transpose(x, 0, 1)
        # x = batch, seqlength, 31,31
        # seqlength = in_channel in that case

        features = torch.flatten(self.cnn(x), start_dim=1)
        # features = (batch, features_size)

        return self.classifier(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts the globule sequences from the mat files."
    )
    parser.add_argument(
        "path", type=str, help="Path to the directory or file to process."
    )

    parser.add_argument("-b", "--batch", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "-ml", "--maxlength", type=int, default=20, help="Maximum length of a sequence."
    )
    parser.add_argument(
        "-d",
        "--downsampling",
        type=str,
        default="uniform",
        choices=["uniform", "similarity"],
        help="Downsampling method to use.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="recurrent",
        choices=["recurrent", "fixedconv"],
        help="Model to use",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="cleaning",
        choices=["cleaning", "classification"],
        help="Model to use",
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=20, help="Number of epochs to run."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=80085, help="Number of epochs to run."
    )
    args = parser.parse_args()

    data_path = args.path

    batch_size = args.batch
    sampling_method = args.downsampling  # Options: uniform, similarity
    max_length = args.maxlength
    _model = args.model  # Options: fixedconv, recurrent
    task = args.task  # Options: cleaning, classification
    epoch = args.epoch

    seed = args.seed
    device = "cuda"

    basename = (
        str(5000)
        + "_"
        + "_".join(
            [
                task,
                str(_model),
                str(sampling_method),
                str(max_length),
                str(batch_size),
                str(epoch),
            ]
        )
        + "-"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logdir = "logs/" + basename

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    import datetime

    if task == "cleaning":
        class_names = ["Reliable", "Unreliable"]
    else:
        class_names = ["Sick", "Healthy"]

    """We use the pytorch lib catalyst to train our models, it takes care of the training loop logics and all dirty things."""

    from catalyst.dl import (
        SupervisedRunner,
        AccuracyCallback,
        AUCCallback,
        ConfusionMatrixCallback,
        PrecisionRecallF1SupportCallback,
    )

    if _model == "recurrent":
        model = RecurrentConvNet(n_classes=2, device=device, softmax=False).to(device)
    else:
        model = FixedSizeConvnet(n_classes=2, device=device, softmax=False).to(device)

    from torch.utils.data import DataLoader, random_split

    """We load the dataset we need to train our model"""

    trainset = GlobulesDataset(
        data_path + "/train",
        preprocessing_method=sampling_method,
        maxlength=max_length,
        task=task,
    )
    valset = GlobulesDataset(
        data_path + "/validation",
        preprocessing_method=sampling_method,
        maxlength=max_length,
        task=task,
        unreliable_downsampling=False,
    )
    testset = GlobulesDataset(
        data_path + "/test",
        preprocessing_method=sampling_method,
        maxlength=max_length,
        task=task,
        unreliable_downsampling=False,
    )

    weights = (len(trainset) - trainset.class_counts()) / len(trainset)
    print("class count", trainset.class_counts())
    print(weights)

    # Sampler used to balance the dataset
    # It will sample more often elements from an under represented class
    print(len(trainset))
    print(trainset.get_class(5))
    print([weights[trainset.get_class(i)] for i in range(len(trainset))][:100])
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        [weights[trainset.get_class(i)] for i in range(len(trainset))], len(trainset)
    )
    print("valset length", len(valset))

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: pad_collate(x, l=max_length),
        num_workers=4,
    )
    valid_loader = DataLoader(
        valset,
        batch_size=batch_size,
        collate_fn=lambda x: pad_collate(
            x,
            l=max_length,
        ),
        num_workers=4,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=lambda x: pad_collate(x, l=max_length),
        num_workers=4,
    )

    from torch.optim import Adam

    # We use categorical cross entropy but on log softmax
    criterion = nn.CrossEntropyLoss()
    # Adam as optimizer since it is the most used optimizer.
    # We'll see later on if other method works better since we know
    # that adam do not generalize well on image processing problem
    optimizer = Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # We use Catalyst to ease the training process and have well saved record in tensorboard
    runner = SupervisedRunner()

    loaders = {"train": train_loader, "valid": valid_loader}

    # Commented out IPython magic to ensure Python compatibility.
    # %load_ext tensorboard
    # %tensorboard --logdir logs/

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=epoch,
        verbose=True,
        valid_metric="f1/class_00",
        valid_loader="valid",
        minimize_valid_metric=False,
        check=True,
        callbacks=[
            PrecisionRecallF1SupportCallback(
                "logits",
                "targets",
                num_classes=2,
                log_on_batch=False,
            ),
            AccuracyCallback("logits", "targets", num_classes=2),
            ConfusionMatrixCallback(
                "logits",
                "targets",
                num_classes=2,
                class_names=class_names,
                normalized=True,
            ),
        ],
    )

    """## Results Summary

    ### Summary on the validation set
    """

    from catalyst.engines.torch import DeviceEngine
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        confusion_matrix,
        recall_score,
        f1_score,
    )

    y_probas = []
    import pickle as pk

    print("valset length", len(valset))

    for pred in runner.predict_loader(
        loader=((x.to("cuda"), y.to("cuda")) for x, y in valid_loader),
        model=model,
        engine=DeviceEngine(device="cuda"),
    ):
        y_probas.append(pred["logits"].cpu().detach().numpy())

    y_probas = np.concatenate(y_probas)

    print(y_probas.shape)

    y_pred = np.argmax(y_probas, axis=1)
    y_true = [y for _, y in valset]

    print("ytrue", len(y_true))

    weights = (len(valset) - valset.class_counts()) / len(valset)

    summary = {"y_proba": y_probas, "y_pred": y_pred, "y_true": y_true}
    summary["params"] = {
        "batch_size": batch_size,
        "sampling_method": sampling_method,
        "max_length": max_length,
        "model": _model,
        "task": task,
    }
    summary["labels"] = class_names
    summary["accuracy"] = accuracy_score(y_true, y_pred)
    summary["precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=[weights[valset.get_class(i)] for i in range(len(valset))],
    )
    summary["recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=[weights[valset.get_class(i)] for i in range(len(valset))],
    )
    summary["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    weights = (len(testset) - testset.class_counts()) / len(testset)
    summary["f1_score"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=[weights[valset.get_class(i)] for i in range(len(valset))],
    )
    summary["f1_score_raw"] = f1_score(y_true, y_pred, average=None)

    with open(basename + "val", "wb") as f:
        pk.dump(summary, f)

    """### Summary on the test set"""

    y_probas = []

    for pred in runner.predict_loader(
        loader=((x.to("cuda"), y.to("cuda")) for x, y in test_loader),
        model=model,
        engine=DeviceEngine(device="cuda"),
    ):
        y_probas.append(pred["logits"].cpu().detach().numpy())

    y_probas = np.concatenate(y_probas)
    y_pred = np.argmax(y_probas, axis=1)
    y_true = [y for _, y in testset]

    weights = (len(testset) - testset.class_counts()) / len(testset)

    summary = {"y_proba": y_probas, "y_pred": y_pred, "y_true": y_true}
    summary["params"] = {
        "batch_size": batch_size,
        "sampling_method": sampling_method,
        "max_length": max_length,
        "model": _model,
        "task": task,
    }
    summary["labels"] = class_names
    summary["accuracy"] = accuracy_score(y_true, y_pred)
    summary["precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=[weights[testset.get_class(i)] for i in range(len(testset))],
    )
    summary["recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=[weights[testset.get_class(i)] for i in range(len(testset))],
    )
    summary["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    weights = (len(testset) - testset.class_counts()) / len(testset)
    summary["f1_score"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        sample_weight=[weights[testset.get_class(i)] for i in range(len(testset))],
    )
    summary["f1_score_raw"] = f1_score(y_true, y_pred, average=None)

    with open(basename + "test", "wb") as f:
        pk.dump(summary, f)
