# JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction
Python implementation of Convolutional Neural Networks (CNNs) proposed in paper

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the JointNet implemented in the paper 
'JointNet: A Common Neural Network for Road and Building Extraction' by Zhang Z., Wang Y. (2019).

The binary focal loss function used in this repository is borrowed from umbertogriffo's focal-loss-keras github repository, which can 
be found at https://github.com/umbertogriffo/focal-loss-keras

The group normalization implementation in Keras used in this repository is the exact same class object defined in the group_norm.py 
file located in titu1994's Keras-Group-Normalization github repository at https://github.com/titu1994/Keras-Group-Normalization.
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



The JointNet model is trained on the training dataset provided by INRIA under the Inria Aerial Image Labeling Dataset (https://project.inria.fr/aerialimagelabeling/), For the inference stage, the raw image is fed to the model as input, and the model output is then thresholded using a threshold value of 0.6. It should be noted that there is no further post - processing of the model output (in order to illustrate the generalization power of the JointNet model), and it is believed that further post - processing would be able to further improve the results.


 - Bellingham Test Image
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/bellingham15.jpg)
 
 - Bellingham Prediction Results
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/bellingham15_output.jpg)
 
 - Bloomington Test Image
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/bloomington17.jpg)
 
 - Bloomington Prediction Results
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/bloomington17_output.jpg)
 
 - Innsbruck Test Image
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/innsbruck21.jpg)
 
 - Innsbruck Prediction Results
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/innsbruck21_output.jpg)
 
 - San Francisco Test Image
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/sfo36.jpg)
 
 - San Francisco Prediction Results
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/sfo36_output.jpg)
 
 - Eastern Tyrol Test Image
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/tyrol-e35.jpg)
 
 - Eastern Tyrol Prediction Results
 ![alt text](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction/blob/master/Test%20Images%20and%20Results/tyrol-e35_output.jpg)
