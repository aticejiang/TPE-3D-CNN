#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:34:49 2018

@author: jiang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import tensorflow as tf

import collections
from sklearn import metrics, preprocessing

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

# =============================================================================
# def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
#     labels_loc = {}
#     train = {}
#     test = {}
#     m = max(groundTruth)
#     for i in range(m):
#         indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
#         np.random.shuffle(indices)
#         labels_loc[i] = indices
#         nb_val = int(proptionVal * len(indices))
#         train[i] = indices[:-nb_val]
#         test[i] = indices[-nb_val:]
# #    whole_indices = []
#     train_indices = []
#     test_indices = []
#     for i in range(m):
# #        whole_indices += labels_loc[i]
#         train_indices += train[i]
#         test_indices += test[i]
#     np.random.shuffle(train_indices)
#     np.random.shuffle(test_indices)
#     return train_indices, test_indices
# =============================================================================
    
def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    validation = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_vallidation = int(proptionVal * len(indices))
        test[i] = indices[:nb_vallidation]
        validation[i] = indices[nb_vallidation:2*nb_vallidation]
        train[i] = indices[2*nb_vallidation:]
#    whole_indices = []
    train_indices = []
    validation_indices=[]
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        validation_indices += validation[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(validation_indices)
    np.random.shuffle(test_indices)
    return train_indices, validation_indices, test_indices

def get_data_name(var):
    return {
            'PaviaU': {'data': 'paviaU', 'gt': 'paviaU_gt', 'category': 9, 'batch_size': 16},
            'Indian_Pines': {'data': 'indian_pines_corrected', 'gt': 'indian_pines_gt', 'category': 16, 'batch_size': 8},
            'KSC': {'data': 'KSC', 'gt': 'KSC_gt', 'category': 13, 'batch_size': 8},
            'Salinas': {'data': 'salinas_corrected', 'gt': 'salinas_gt', 'category': 16, 'batch_size': 8},
            'PaviaC': {'data': 'pavia', 'gt': 'pavia_gt', 'category': 9, 'batch_size': 16},
            'Botswana': {'data': 'Botswana', 'gt': 'Botswana_gt', 'category': 14, 'batch_size': 16}
    }.get(var,'error')

def data_preprocessing(image, ground_truth, patch_length, validation_rate, data_name, interp = False):
    img_width, img_height = 2* patch_length + 1, 2* patch_length + 1
    
    _data = sio.loadmat(image)
    _gt = sio.loadmat(ground_truth)
    data_IN = _data[get_data_name(data_name)['data']]
    gt_IN = _gt[get_data_name(data_name)['gt']]
    new_gt_IN = gt_IN
    
    if interp == True:
        data_inter = np.transpose(data_IN,(2,0,1))
        data_tmp = tf.image.resize_images(data_inter.astype(np.float32), [50,data_IN.shape[0]], method = 1)
        sess = tf.Session()
        with sess.as_default():
            data_tmp = data_tmp.eval()
        data_IN = np.transpose(data_tmp,(1,2,0))
    
    data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
    
    data = preprocessing.scale(data)
    
    data = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
    #whole_data = data
    #padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)
    #data_zeropadding = ZeroPadding3D(padding=(PATCH_LENGTH,PATCH_LENGTH,0), name = 'ZeroPadding')(data)
    data_zeropadding = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)),
               'constant', constant_values=0)
    
    train_indices, validation_indices, test_indices = sampling(validation_rate, gt)
    
    train_data = np.zeros((len(train_indices), img_width, img_height, data_IN.shape[2]))
    validation_data = np.zeros((len(validation_indices), img_width, img_height, data_IN.shape[2]))
    test_data = np.zeros((len(test_indices), img_width, img_height, data_IN.shape[2]))
    
    y_train = gt[train_indices] -1
    y_train = to_categorical(np.asarray(y_train))
    
    y_validation = gt[validation_indices] -1
    y_validation = to_categorical(np.asarray(y_validation))
    
    y_test = gt[test_indices] -1
    y_test = to_categorical(np.asarray(y_test))
    
    train_assign = indexToAssignment(train_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(data_zeropadding,train_assign[i][0],train_assign[i][1],patch_length)
    
    validation_assign = indexToAssignment(validation_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(validation_assign)):
        validation_data[i] = selectNeighboringPatch(data_zeropadding,validation_assign[i][0],validation_assign[i][1],patch_length)
        
    test_assign = indexToAssignment(test_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(data_zeropadding,test_assign[i][0],test_assign[i][1],patch_length)
    
    train_data = np.expand_dims(train_data, axis=4)
    validation_data = np.expand_dims(validation_data, axis=4)
    test_data = np.expand_dims(test_data, axis=4)
    
    print('train_data_shape:', train_data.shape, 'train_labels_shape:', y_train.shape)
    print('validation_data_shape:', validation_data.shape)
    print('test_data_shape:', test_data.shape)
    
    return train_data, y_train, validation_data, y_validation, test_data, y_test



#flag 
#    data_inter = np.transpose(data_IN,(2,0,1))
#    data_tmp = tf.image.resize_images(data_inter.astype(np.float32), [50,610], method = 1)
#    sess = tf.Session()
#    with sess.as_default():
#        data_tmp = data_tmp.eval()
#    data_IN = np.transpose(data_tmp,(1,2,0))
def data_preprocessing_inter(image, ground_truth, patch_length, validation_rate, data_name):
    img_width, img_height = 2* patch_length + 1, 2* patch_length + 1
    
    _data = sio.loadmat(image)
    _gt = sio.loadmat(ground_truth)
    data_IN = _data[get_data_name(data_name)['data']]
    gt_IN = _gt[get_data_name(data_name)['gt']]
    new_gt_IN = gt_IN
    
    data_inter = np.transpose(data_IN,(2,0,1))
    data_tmp = tf.image.resize_images(data_inter.astype(np.float32), [50,610], method = 1)
    sess = tf.Session()
    with sess.as_default():
        data_tmp = data_tmp.eval()
    data_IN = np.transpose(data_tmp,(1,2,0))
    
    data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
    
    data = preprocessing.scale(data)
    
    data = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])
    #whole_data = data
    #padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)
    #data_zeropadding = ZeroPadding3D(padding=(PATCH_LENGTH,PATCH_LENGTH,0), name = 'ZeroPadding')(data)
    data_zeropadding = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)),
               'constant', constant_values=0)
    
    train_indices, validation_indices, test_indices = sampling(validation_rate, gt)
    
    train_data = np.zeros((len(train_indices), img_width, img_height, data_IN.shape[2]))
    validation_data = np.zeros((len(validation_indices), img_width, img_height, data_IN.shape[2]))
    test_data = np.zeros((len(test_indices), img_width, img_height, data_IN.shape[2]))
    
    y_train = gt[train_indices] -1
    y_train = to_categorical(np.asarray(y_train))
    
    y_validation = gt[validation_indices] -1
    y_validation = to_categorical(np.asarray(y_validation))
    
    y_test = gt[test_indices] -1
    y_test = to_categorical(np.asarray(y_test))
    
    train_assign = indexToAssignment(train_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(data_zeropadding,train_assign[i][0],train_assign[i][1],patch_length)
    
    validation_assign = indexToAssignment(validation_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(validation_assign)):
        validation_data[i] = selectNeighboringPatch(data_zeropadding,validation_assign[i][0],validation_assign[i][1],patch_length)
        
    test_assign = indexToAssignment(test_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(data_zeropadding,test_assign[i][0],test_assign[i][1],patch_length)
    
    train_data = np.expand_dims(train_data, axis=4)
    validation_data = np.expand_dims(validation_data, axis=4)
    test_data = np.expand_dims(test_data, axis=4)
    
    print('train_data_shape:', train_data.shape, 'train_labels_shape:', y_train.shape)
    print('validation_data_shape:', validation_data.shape)
    print('test_data_shape:', test_data.shape)
    
    return train_data, y_train, validation_data, y_validation, test_data, y_test