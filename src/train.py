from dataset import GlobulesDatasetCleaning, pad_collate, GlobulesDatasetUsable
from model import RecurrentConvNet, FixedSizeConvnet
from sampler import BalancedBatchSampler
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from torch.utils.data import DataLoader, random_split

import pickle as pk

from torch.optim import Adam
import torch
import torch.nn as nn

from catalyst.dl import SupervisedRunner, AccuracyCallback, AUCCallback, ConfusionMatrixCallback, \
    PrecisionRecallF1ScoreCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts the globule sequences from the mat files.')
    parser.add_argument('training', type=str, help='Path to the file containing the training set.')
    parser.add_argument('testing', type=str, default=None, help='Path to the file containing the testing set.')
    parser.add_argument('summary', type=str, help='Path to where store the results, without extension')
    parser.add_argument('-v', '--validation', type=str, default=False,
                        help='Path to the file containing the validation set.')

    parser.add_argument('-d', '--dataset', type=str, choices=['cleaning', 'classification', 'allinone'],
                        default='cleaning',
                        help='Which mode to use, cleaning preprocess the data to be used to train the datacleaner only, '
                             'classification the data used for healthy/sick classification'
                             'allinone make 3 classes model in one stage.')

    parser.add_argument('-s', '--seqlength', type=int, default=-1, help='If < 0 it uses a recurrent convnet with not '
                                                                        'fixed length, if set to k, it uses sequences of '
                                                                        'size max k and pad with black images'
                                                                        ' sequences that are shorter')
    parser.add_argument('-e', '--epochs', type=int, default=128, help='Number of epoch to make.')
    parser.add_argument('-b', '--batch', type=int, default=128, help='Size of a minibatch')

    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    model_date = time.time()

    path = args.training

    batch_size = args.batch
    num_epoch = args.epochs

    classes_name = ['Healthy', 'Sick']

    modelname = 'rnn' if args.seqlength < 0 else 'cnn' + str(args.seqlength)


    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    if args.dataset == 'cleaning':
        n_classes = 2
        # Healthy/sick vs blurry, non usable etc...
        classes_name = ['Reliable', 'Unreliable']
        DatasetClass = GlobulesDatasetCleaning
        name =  str(args.training) + 'cleaning'
    elif args.dataset == 'classification':
        n_classes = 2
        classes_name = ['Healthy', 'Sick']
        DatasetClass = GlobulesDatasetUsable
        name = str(args.training) + 'classification'
    else:
        n_classes = 3
        classes_name = ['Healthy', 'Sick', 'Unreliable']
        DatasetClass = GlobulesDatasetCleaning
        name = modelname + '-' + '3classes'

    logdir = 'logs/' + modelname + name

    # Loaging of our preprocessed dataset
    train_dataset = DatasetClass(path, n_classes=n_classes)

    if args.validation:
        validation_dataset = DatasetClass(args.validation, n_classes=n_classes)

    # split into train and validation 70%, 30%
    else:
        train_dataset, validation_dataset = random_split(train_dataset, lengths=[int(len(train_dataset) * 0.7),
                                                                                 len(train_dataset) - int(
                                                                                     len(train_dataset) * 0.7)])


    # We use a sampler to get balanced batches
    train_sampler = BalancedBatchSampler(train_dataset, [x[1] for x in train_dataset])

    # We instantiate loaders with padding to make all sequences the same length
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=lambda x: pad_collate(x, l=args.seqlength))
    valid_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=lambda x: pad_collate(x, l=args.seqlength))

    loaders = {"train": train_loader, "valid": valid_loader}

    print("RUN")

    if args.seqlength < 0:
        model = RecurrentConvNet(n_classes=n_classes, device=device, softmax=False).to(device)
    else:
        model = FixedSizeConvnet(n_classes=n_classes, in_channels=args.seqlength, device=device)

    # We use categorical cross entropy but on log softmax
    criterion = nn.CrossEntropyLoss()
    # Adam as optimizer since it is the most used optimizer.
    # We'll see later on if other method works better since we know
    # that adam do not generalize well on image processing problem
    optimizer = Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # We use Catalyst to ease the training process and have well saved record in tensorboard
    runner = SupervisedRunner(device=device)

    # We run the training loop with Accuracy metric and confusion matrix to get recall data.
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epoch,
        verbose=True,
        callbacks=[AccuracyCallback(num_classes=n_classes)]
    )

    if args.testing is not None:
        testset = DatasetClass(args.testing, n_classes=n_classes)
        testset_loader = DataLoader(testset,
                                    batch_size=batch_size,
                                    collate_fn=lambda x: pad_collate(x, l=args.seqlength),
                                    shuffle=False)

        y_probas = runner.predict_loader(loader=testset_loader,
                                         model=model,
                                         resume=logdir + "/" + "checkpoints/best.pth")
        y_pred = np.argmax(y_probas, axis=1)
        y_true = testset.y

        summary = {"y_proba": y_probas, "y_pred": y_pred, "y_true": y_true}
        summary['params'] = {"training_set": args.training, }
        summary['labels'] = classes_name
        summary["accuracy"] = accuracy_score(y_true, y_pred)
        summary["precision"] = precision_score(y_true, y_pred, average='micro')
        summary["recall"] = recall_score(y_true, y_pred, average='micro')
        summary["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        summary["f1_score"] = f1_score(y_true, y_pred, average='micro')

        with open(args.summary, 'wb') as f:
            pk.dump(summary, f)


    y_probas = runner.predict_loader(loader=valid_loader,
                                     model=model,
                                     resume=logdir + "/" + "checkpoints/best.pth")
    y_pred = np.argmax(y_probas, axis=1)
    y_true = validation_dataset.y

    summary = {"y_proba": y_probas, "y_pred": y_pred, "y_true": y_true}
    summary['params'] = {"training_set": args.training, }
    summary['labels'] = classes_name
    summary["accuracy"] = accuracy_score(y_true, y_pred)
    summary["precision"] = precision_score(y_true, y_pred, average='micro')
    summary["recall"] = recall_score(y_true, y_pred, average='micro')
    summary["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    summary["f1_score"] = f1_score(y_true, y_pred, average='micro')

    with open(args.summary + "val", 'wb') as f:
        pk.dump(summary, f)