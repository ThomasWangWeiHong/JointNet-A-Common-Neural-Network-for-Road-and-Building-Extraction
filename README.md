# JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction
Python implementation of Convolutional Neural Networks (CNNs) proposed in paper

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the JointNet implemented in the paper 
'JointNet: A Common Neural Network for Road and Building Extraction' by Zhang Z., Wang Y. (2019).

The binary focal loss function used in this repository is borrowed from umbertogriffo's focal-loss-keras github repository, which can 
be found at https://github.com/umbertogriffo/focal-loss-keras

The group normalization implementation in Keras used in this repository is the exact same class object defined in the group_norm.py 
file located in titu1994's Keras-Group-Normalization github repository at https://github.com/titu1994/Keras-Group-Normalization
Please ensure that the group_norm.py file is placed in the correct directory before use.

Requirements:

- cv2
- glob
- json
- numpy
- rasterio
- tensorflow
- group_norm (downloaded from https://github.com/titu1994/Keras-Group-Normalization)
- keras (with tensorflow backend)
