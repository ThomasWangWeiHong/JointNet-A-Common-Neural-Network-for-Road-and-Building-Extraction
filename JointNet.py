import cv2
import glob
import json
import numpy as np
import rasterio
import tensorflow as tf
from group_norm import GroupNormalization
from keras import backend as K
from keras.models import Input, Model
from keras.layers import concatenate, Conv2D, UpSampling2D
from keras.optimizers import Adam



def training_mask_generation(input_image_filename, input_geojson_filename):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        image = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        
    mask = np.zeros((image.shape[0], image.shape[1]))
    
    ulx = metadata['transform'][2]
    xres = metadata['transform'][0]
    uly = metadata['transform'][5]
    yres = metadata['transform'][4]
                                      
    lrx = ulx + (image.shape[1] * xres)                                                         
    lry = uly - (image.shape[0] * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((image.shape[1]) ** 2 / (image.shape[1] + 1)) / (lrx - ulx)
        yf = ((image.shape[0]) ** 2 / (image.shape[0] + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask, position, 1)
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, image_height_size, image_width_size, mode, percentage_overlap, 
                                      buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image and its mask in the form of fixed size segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((image_array.shape[0], x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = mask_array
        
    img_list = []
    mask_list = []
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, 0]
            mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
            mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
            mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
            mask_flip_hor = cv2.flip(mask_original, 0)
            mask_flip_vert = cv2.flip(mask_original, 1)
            mask_flip_both = cv2.flip(mask_original, -1)
            mask_list.extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, mask_flip_vert, 
                              mask_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(mask_list), image_height_size, image_width_size, 1))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        mask_segment_array[index, :, :, 0] = mask_list[index]
        
    return image_segment_array, mask_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\' + 'Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Training Polygons' + '\\Train_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    
    for file in range(len(img_files)):
        with rasterio.open(img_files[file]) as f:
            metadata = f.profile
            img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
            
        mask = training_mask_generation(img_files[file], polygon_files[file])
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 0, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 1, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 2, 
                                                                      percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 3, 
                                                                      percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    
    return img_full_array, mask_full_array



def binary_focal_loss(gamma = 2, alpha = 0.25):
    """
    Code borrowed from https://github.com/umbertogriffo/focal-loss-keras/blob/master/losses.py
    
    Binary form of focal loss.
      FL(p_t) = - alpha * (1 - p_t) ** gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
      
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
     
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed



def dice_coef(y_true, y_pred):
    """ 
    This function generates the dice coefficient for use in semantic segmentation model training. 
    
    """
    
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    coef = (2 * intersection) / (K.sum(y_true_flat) + K.sum(y_pred_flat))
    
    return coef



def dice_coef_loss(y_true, y_pred):
    """ 
    This function generates the dice coefficient loss function for use in semantic segmentation model training. 
    
    """
    
    return -dice_coef(y_true, y_pred)



def JointNet(img_height_size = 512, img_width_size = 512, n_bands = 3, growth_rate_1 = 32, growth_rate_2 = 64, 
             growth_rate_3 = 128, growth_rate_net_bridge = 256, group_filters = 8, l_r = 0.0001):
    """
    This function is used to generate the JointNet architecture as described in the paper 'JointNet: A Common Neural Network
    for Road and Building Extraction' by Zhang Z., Wang Y. (2019).
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - growth_rate_1: Number of filters to be used for each atrous convolution in the first dense atrous convolution block
    - growth_rate_2: Number of filters to be used for each atrous convolution in the second dense atrous convolution block
    - growth_rate_3: Number of filters to be used for each atrous convolution in the third dense atrous convolution block
    - growth_rate_net_bridge: Number of filters to be used for each atrous convolution in the network bridge
    - group_filters: Number of groups to be used for group normalization
    - l_r: Learning rate to be applied for the Adam optimizer
    
    Outputs:
    -
    
    """
    
    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    
    
    ec_1_ac_1 = Conv2D(growth_rate_1, (3, 3), padding = 'same', activation = 'relu')(img_input)
    ec_1_ac_1_gn = GroupNormalization(groups = group_filters, axis = -1, 
                                      epsilon = 0.1)(ec_1_ac_1)
    ec_1_ac_1_output = concatenate([img_input, ec_1_ac_1_gn])
    ec_1_ac_2 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(ec_1_ac_1_output)
    ec_1_ac_2_gn = GroupNormalization(groups = group_filters, axis = -1, 
                                      epsilon = 0.1)(ec_1_ac_2)
    ec_1_ac_2_output = concatenate([img_input, ec_1_ac_1_gn, ec_1_ac_2_gn])
    ec_1_ac_3 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(ec_1_ac_2_output)
    ec_1_ac_3_gn = GroupNormalization(groups = group_filters, axis = -1, 
                                      epsilon = 0.1)(ec_1_ac_3)
    ec_1_ac_3_output = concatenate([img_input, ec_1_ac_1_gn, ec_1_ac_2_gn, ec_1_ac_3_gn])
    ec_1_ac_4 = Conv2D(growth_rate_1, (3, 3), padding = 'same', activation = 'relu')(ec_1_ac_3_output)
    ec_1_ac_4_gn = GroupNormalization(groups = group_filters, axis = -1, 
                                      epsilon = 0.1)(ec_1_ac_4)
    ec_1_ac_4_output = concatenate([img_input, ec_1_ac_1_gn, ec_1_ac_2_gn, ec_1_ac_3_gn, ec_1_ac_4_gn])
    ec_1_ac_5 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(ec_1_ac_4_output)
    ec_1_ac_5_gn = GroupNormalization(groups = group_filters, axis = -1, 
                                      epsilon = 0.1)(ec_1_ac_5)
    ec_1_ac_5_output = concatenate([img_input, ec_1_ac_1_gn, ec_1_ac_2_gn, ec_1_ac_3_gn, ec_1_ac_4_gn, ec_1_ac_5_gn])
    ec_1_ac_6 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(ec_1_ac_5_output)
    ec_1_ac_6_gn = GroupNormalization(groups = group_filters, axis = -1, 
                                      epsilon = 0.1)(ec_1_ac_6)
    ec_1_ac_6_output = concatenate([img_input, ec_1_ac_1_gn, ec_1_ac_2_gn, ec_1_ac_3_gn, ec_1_ac_4_gn, ec_1_ac_5_gn, 
                                    ec_1_ac_6_gn])
    
    ec_1_DO = Conv2D(int(4 * growth_rate_1), (1, 1), padding = 'same', activation = 'relu')(ec_1_ac_6_output)
    ec_1_RO = Conv2D(growth_rate_1, (1, 1), strides = (2, 2), padding = 'same', activation = 'relu')(img_input)
    
    
    ec_2_ac_1 = Conv2D(growth_rate_2, (3, 3), padding = 'same', activation = 'relu')(ec_1_RO)
    ec_2_ac_1_gn = GroupNormalization(groups = int(group_filters * 2), axis = -1, 
                                      epsilon = 0.1)(ec_2_ac_1)
    ec_2_ac_1_output = concatenate([ec_1_RO, ec_2_ac_1_gn])
    ec_2_ac_2 = Conv2D(growth_rate_2, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(ec_2_ac_1_output)
    ec_2_ac_2_gn = GroupNormalization(groups = int(group_filters * 2), axis = -1, 
                                      epsilon = 0.1)(ec_2_ac_2)
    ec_2_ac_2_output = concatenate([ec_1_RO, ec_2_ac_1_gn, ec_2_ac_2_gn])
    ec_2_ac_3 = Conv2D(growth_rate_2, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(ec_2_ac_2_output)
    ec_2_ac_3_gn = GroupNormalization(groups = int(group_filters * 2), axis = -1, 
                                      epsilon = 0.1)(ec_2_ac_3)
    ec_2_ac_3_output = concatenate([ec_1_RO, ec_2_ac_1_gn, ec_2_ac_2_gn, ec_2_ac_3_gn])
    ec_2_ac_4 = Conv2D(growth_rate_2, (3, 3), padding = 'same', activation = 'relu')(ec_2_ac_3_output)
    ec_2_ac_4_gn = GroupNormalization(groups = int(group_filters * 2), axis = -1, 
                                      epsilon = 0.1)(ec_2_ac_4)
    ec_2_ac_4_output = concatenate([ec_1_RO, ec_2_ac_1_gn, ec_2_ac_2_gn, ec_2_ac_3_gn, ec_2_ac_4_gn])
    ec_2_ac_5 = Conv2D(growth_rate_2, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(ec_2_ac_4_output)
    ec_2_ac_5_gn = GroupNormalization(groups = int(group_filters * 2), axis = -1, 
                                      epsilon = 0.1)(ec_2_ac_5)
    ec_2_ac_5_output = concatenate([ec_1_RO, ec_2_ac_1_gn, ec_2_ac_2_gn, ec_2_ac_3_gn, ec_2_ac_4_gn, ec_2_ac_5_gn])
    ec_2_ac_6 = Conv2D(growth_rate_2, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(ec_2_ac_5_output)
    ec_2_ac_6_gn = GroupNormalization(groups = int(group_filters * 2), axis = -1, 
                                      epsilon = 0.1)(ec_2_ac_6)
    ec_2_ac_6_output = concatenate([ec_1_RO, ec_2_ac_1_gn, ec_2_ac_2_gn, ec_2_ac_3_gn, ec_2_ac_4_gn, ec_2_ac_5_gn, 
                                    ec_2_ac_6_gn])
    
    ec_2_DO = Conv2D(int(4 * growth_rate_2), (1, 1), padding = 'same', activation = 'relu')(ec_2_ac_6_output)
    ec_2_RO = Conv2D(growth_rate_2, (1, 1), strides = (2, 2), padding = 'same', activation = 'relu')(ec_1_RO)
    
    
    ec_3_ac_1 = Conv2D(growth_rate_3, (3, 3), padding = 'same', activation = 'relu')(ec_2_RO)
    ec_3_ac_1_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(ec_3_ac_1)
    ec_3_ac_1_output = concatenate([ec_2_RO, ec_3_ac_1_gn])
    ec_3_ac_2 = Conv2D(growth_rate_3, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(ec_3_ac_1_output)
    ec_3_ac_2_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(ec_3_ac_2)
    ec_3_ac_2_output = concatenate([ec_2_RO, ec_3_ac_1_gn, ec_3_ac_2_gn])
    ec_3_ac_3 = Conv2D(growth_rate_3, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(ec_3_ac_2_output)
    ec_3_ac_3_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(ec_3_ac_3)
    ec_3_ac_3_output = concatenate([ec_2_RO, ec_3_ac_1_gn, ec_3_ac_2_gn, ec_3_ac_3_gn])
    ec_3_ac_4 = Conv2D(growth_rate_3, (3, 3), padding = 'same', activation = 'relu')(ec_3_ac_3_output)
    ec_3_ac_4_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(ec_3_ac_4)
    ec_3_ac_4_output = concatenate([ec_2_RO, ec_3_ac_1_gn, ec_3_ac_2_gn, ec_3_ac_3_gn, ec_3_ac_4_gn])
    ec_3_ac_5 = Conv2D(growth_rate_3, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(ec_3_ac_4_output)
    ec_3_ac_5_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(ec_3_ac_5)
    ec_3_ac_5_output = concatenate([ec_2_RO, ec_3_ac_1_gn, ec_3_ac_2_gn, ec_3_ac_3_gn, ec_3_ac_4_gn, ec_3_ac_5_gn])
    ec_3_ac_6 = Conv2D(growth_rate_3, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(ec_3_ac_5_output)
    ec_3_ac_6_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(ec_3_ac_6)
    ec_3_ac_6_output = concatenate([ec_2_RO, ec_3_ac_1_gn, ec_3_ac_2_gn, ec_3_ac_3_gn, ec_3_ac_4_gn, ec_3_ac_5_gn, 
                                    ec_3_ac_6_gn])
    
    ec_3_DO = Conv2D(int(4 * growth_rate_3), (1, 1), padding = 'same', activation = 'relu')(ec_3_ac_6_output)
    ec_3_RO = Conv2D(growth_rate_3, (1, 1), strides = (2, 2), padding = 'same', activation = 'relu')(ec_2_RO)
    
    
    net_bridge = Conv2D(growth_rate_net_bridge, (1, 1), padding = 'same', activation = 'relu')(ec_3_RO)
    net_bridge_up = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(net_bridge)
    
    
    dc_3_input = concatenate([net_bridge_up, ec_3_DO])
    dc_3_output = Conv2D(growth_rate_3, (1, 1), padding = 'same', activation = 'relu')(dc_3_input)
    dc_3_up = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(dc_3_output)
    
    
    dc_2_input = concatenate([dc_3_up, ec_2_DO])
    dc_2_output = Conv2D(growth_rate_2, (1, 1), padding = 'same', activation = 'relu')(dc_2_input)
    dc_2_up = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(dc_2_output)
    
    
    dc_1_input = concatenate([dc_2_up, ec_1_DO])
    dc_1_ac_1 = Conv2D(growth_rate_1, (3, 3), padding = 'same', activation = 'relu')(dc_1_input)
    dc_1_ac_1_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(dc_1_ac_1)
    dc_1_ac_1_output = concatenate([dc_1_input, dc_1_ac_1_gn])
    dc_1_ac_2 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(dc_1_ac_1_output)
    dc_1_ac_2_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(dc_1_ac_2)
    dc_1_ac_2_output = concatenate([dc_1_input, dc_1_ac_1_gn, dc_1_ac_2_gn])
    dc_1_ac_3 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(dc_1_ac_2_output)
    dc_1_ac_3_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(dc_1_ac_3)
    dc_1_ac_3_output = concatenate([dc_1_input, dc_1_ac_1_gn, dc_1_ac_2_gn, dc_1_ac_3_gn])
    dc_1_ac_4 = Conv2D(growth_rate_1, (3, 3), padding = 'same', activation = 'relu')(dc_1_ac_3_output)
    dc_1_ac_4_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(dc_1_ac_4)
    dc_1_ac_4_output = concatenate([dc_1_input, dc_1_ac_1_gn, dc_1_ac_2_gn, dc_1_ac_3_gn, dc_1_ac_4_gn])
    dc_1_ac_5 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (2, 2), activation = 'relu')(dc_1_ac_4_output)
    dc_1_ac_5_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(dc_1_ac_5)
    dc_1_ac_5_output = concatenate([dc_1_input, dc_1_ac_1_gn, dc_1_ac_2_gn, dc_1_ac_3_gn, dc_1_ac_4_gn, dc_1_ac_5_gn])
    dc_1_ac_6 = Conv2D(growth_rate_1, (3, 3), padding = 'same', dilation_rate = (5, 5), activation = 'relu')(dc_1_ac_5_output)
    dc_1_ac_6_gn = GroupNormalization(groups = int(group_filters * 4), axis = -1, 
                                      epsilon = 0.1)(dc_1_ac_6)
    dc_1_ac_6_output = concatenate([dc_1_input, dc_1_ac_1_gn, dc_1_ac_2_gn, dc_1_ac_3_gn, dc_1_ac_4_gn, dc_1_ac_5_gn, 
                                    dc_1_ac_6_gn])
    
    dc_1_DO = Conv2D(int(4 * growth_rate_1), (1, 1), padding = 'same', activation = 'relu')(dc_1_ac_6_output)
    
    
    class_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(dc_1_DO)
    
    
    jointnet_model = Model(inputs = img_input, outputs = class_layer)
    jointnet_model.compile(loss = dice_coef_loss, optimizer = Adam(lr = l_r), metrics = [dice_coef])
    
    return jointnet_model



def image_model_predict(input_image_filename, output_filename, img_height_size, img_width_size, fitted_model, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for prediction. The 
    output mask is then allocated to its corresponding location in the image in order to obtain the complete mask for the 
    entire image without being constrained by image size. 
    
    Inputs:
    - input_image_filename: File path of image file for which prediction is to be conducted
    - output_filename: File path of output predicted binary raster mask file
    - img_height_size: Height of image patches to be used for model prediction
    - img_height_size: Width of image patches to be used for model prediction
    - fitted_model: Trained keras model which is to be used for prediction
    - write: Boolean indicating whether to write predicted binary raster mask to file
    
    Output:
    - mask_complete: Numpy array of predicted binary raster mask for input image
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
     
    y_size = ((img.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((img.shape[1] // img_width_size) + 1) * img_width_size
    
    if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, img.shape[1], img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((img.shape[0], x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    else:
         img_complete = img
            
    mask = np.zeros((img_complete.shape[0], img_complete.shape[1], 1))
    img_holder = np.zeros((1, img_height_size, img_width_size, img.shape[2]))
    
    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img.shape[2]]
            preds = fitted_model.predict(img_holder)
            mask[i : i + img_height_size, j : j + img_width_size, 0] = preds[0, :, :, 0]
            
    mask_complete = np.expand_dims(mask[0 : img.shape[0], 0 : img.shape[1], 0], axis = 2)
    mask_complete = np.transpose(mask_complete, [2, 0, 1]).astype('float32')
    
    
    if write:
        metadata['count'] = 1
        metadata['dtype'] = 'float32'
        
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(mask_complete)
    
    return mask_complete