## Classification stage

In `src` we present the notebook with data loading and model training routine. We also show performance on validation set.

In `data` we provide the stage 2 sequences images for Tank-treading and Flipping cells (output from stage 1 - cleaning).

Moreover we provide in `pretrained` a pretrained model that can be used directly to classify Tank-treading and Flipping cell sequences.

In order to reproduce the results, please provide the correct paths for data and model as mentioned in the notebook. Note that when loading the model, specify only the top level directory path for the model.
Data directory structure (TT - TankTreading, others - Flipping):
```
Data/
  Train/
    TT/
    others/
  Valid/
    TT/
    others/
  Test/
```
