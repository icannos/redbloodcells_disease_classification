### Code for paper Classification of Red Cell Dynamics with Convolutional and Recurrent Neural Networks: A Sickle Cell Disease Case Study (Under Review).

### Abstract

The fraction of red blood cells adopting a specific motion under low shear flow is a promising inexpensive marker for monitoring the clinical status of patients with sickle cell disease. Its high-throughput measurement relies on the video analysis of thousands of cell motions for each blood sample to eliminate a large majority of unreliable samples (out of focus or overlapping cells) and discriminate between tank-treading and flipping motion, characterizing highly and poorly deformable cells respectively. Moreover, these videos are of different durations (from 6 to more than 100 frames). We present a two-stage end-to-end machine learning pipeline able to automatically classify cell motions in videos with a high class imbalance. By extending, comparing, and combining two state-of-the-art methods, a convolutional neural network (CNN) model and a recurrent CNN, we are able to automatically discard 97\% of the unreliable cell sequences (first stage) and classify highly and poorly deformable red cell sequences with 97\% accuracy and an F1-score of 0.94 (second stage). Dataset and codes are publicly released for the community.

### Requirements

```
torch
fastai
numpy
catalyst
```

### Structure

We provide code and demos for both stages as individual directories (`Stage_1` and `Stage_2`). In both case we provide the sources, the pretrained models and sample data. `Stage_1` performs the separation of unreliable samples from the data and `Stage_2` performs the classification of Tank-treading and Flipping cells.
