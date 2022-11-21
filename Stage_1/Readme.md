## Cleaning stage

In `src` we present the source file used to load the data and train the model. 
We wrapped up our dataset into a torch Dataset to unsure easy access to the data.

The `src/train.py` file contains the training loop and the model definition. It performs the preprocessing of 
the data and trains the model.  The model is trained on the GPU if available.

The jupyter notebook provided alongside this readme demonstrates the use of the model in production and gives some
examples of its results.

Moreover we provide in `pretrained` a pretrained model that can be used directly to detect usable and unsable data.