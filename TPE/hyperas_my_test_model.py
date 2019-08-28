#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 00:30:00 2018

@author: jiang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding3D
from keras.layers import BatchNormalization
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.layers.advanced_activations import ELU
#from .imagenet_utils import decode_predictions
#from .imagenet_utils import preprocess_input
#from .imagenet_utils import _obtain_input_shape


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
kernel_Initializer = 'he_normal'  #glorot_uniform
kernel_Regularizer = l2(1.e-3)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), kernel_initializer=kernel_Initializer,
               kernel_regularizer=kernel_Regularizer, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#    x = Activation('relu')(x)
    x = ELU()(x)

    x = Conv3D(filters2, kernel_size, kernel_initializer=kernel_Initializer, kernel_regularizer=kernel_Regularizer,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#    x = Activation('relu')(x)
    x = ELU()(x)

    x = Conv3D(filters3, (1, 1, 1), kernel_initializer=kernel_Initializer,
               kernel_regularizer=kernel_Regularizer, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
#    x = Activation('relu')(x)
    x = ELU()(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1, 1)):   #strides=(2, 2, 2)
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensornb_
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,2
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(filters1, (1, 1, 1), strides=strides, kernel_initializer=kernel_Initializer,
               kernel_regularizer=kernel_Regularizer, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#    x = Activation('relu')(x)
    x = ELU()(x)

    x = Conv3D(filters2, kernel_size, padding='same', kernel_initializer=kernel_Initializer,
               kernel_regularizer=kernel_Regularizer, name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#    x = Activation('relu')(x)
    x = ELU()(x)

    x = Conv3D(filters3, (1, 1, 1), kernel_initializer=kernel_Initializer,
               kernel_regularizer=kernel_Regularizer, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv3D(filters3, (1, 1, 1), strides=strides, kernel_initializer=kernel_Initializer,
                      kernel_regularizer=kernel_Regularizer, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
#    x = Activation('relu')(x)
    x = ELU()(x)
    return x


def ResNet50_3D(weights=None,input_tensor=None, 
                input_shape=None, classes=None):
    """Instantiates the ResNet50_3D architecture.


    # Arguments
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    """
  
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 4
    else:
        bn_axis = 1

#    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv3D(64, (1, 1, 3), strides=(1, 1, 2), kernel_initializer=kernel_Initializer,
               kernel_regularizer=kernel_Regularizer, padding='valid', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#    x = Activation('relu')(x)
    x = ELU()(x)
    x = MaxPooling3D((1, 1, 3), strides=(1, 1, 2))(x)

    x = conv_block(x, (1, 1, 3), [16, 16, 64], stage=2, block='a', strides=(1, 1, 2))  #[64, 64, 256]
    x = identity_block(x, (1, 1, 3), [16, 16, 64], stage=2, block='b')
    x = identity_block(x, (1, 1, 3), [16, 16, 64], stage=2, block='c')

# =============================================================================
#     x = conv_block(x, (1, 1, 3), [32, 32, 128], stage=3, block='a', strides=(1, 1, 2))       #[128, 128, 512]
#     x = identity_block(x, (1, 1, 3), [32, 32, 128], stage=3, block='b')
#     x = identity_block(x, (1, 1, 3), [32, 32, 128], stage=3, block='c')
#     x = identity_block(x, (1, 1, 3), [32, 32, 128], stage=3, block='d')
# 
#     x = conv_block(x, 3, [64, 64, 256], stage=4, block='a')      #[256, 256, 1024]
#     x = identity_block(x, 3, [64, 64, 256], stage=4, block='b')
#     x = identity_block(x, 3, [64, 64, 256], stage=4, block='c')
#     x = identity_block(x, 3, [64, 64, 256], stage=4, block='d')
#     x = identity_block(x, 3, [64, 64, 256], stage=4, block='e')
#     x = identity_block(x, 3, [64, 64, 256], stage=4, block='f')
# =============================================================================
 
    x = conv_block(x, 3, [32, 32, 128], stage=5, block='a')      #[512, 512, 2048] [128, 128, 512]
    x = identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    x = AveragePooling3D((7, 7 ,1), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dropout(0.5055)(x)
#    
#    
    x = Dense(512, kernel_initializer=kernel_Initializer,
              kernel_regularizer=kernel_Regularizer, name='dense')(x)
    x = ELU()(x)
#    
#    
    x = Dense(classes, activation='softmax', kernel_initializer=kernel_Initializer,
              kernel_regularizer=kernel_Regularizer, name='softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50_3D')

    if weights is not None:
        model.load_weights(weights)

    return model