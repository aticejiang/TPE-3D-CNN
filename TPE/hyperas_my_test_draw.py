#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 00:58:47 2018

@author: jiang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
from keras.models import load_model
import tensorflow as tf

from sklearn import metrics, preprocessing
import my_test_new_model
import my_test_newnew

def sampling_all(groundTruth):
    labels_loc = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        labels_loc[i] = indices
    whole_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]
    return whole_indices

def indexToAssignment_raw(index_, Col):
    assign_0 = index_ // Col
    assign_1 = index_ % Col
    return [assign_0, assign_1]

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map, aspect='auto') 
    fig.savefig(savePath, dpi = dpi)

    return 0

def classification_draw(my_dir, image, ground_truth, patch_length, data_name, epochs, index_iter = 1, best_weights_path = None, interp = False):
    _data = sio.loadmat(image)
    _gt = sio.loadmat(ground_truth)
    data_IN = _data[my_test_newnew.get_data_name(data_name)['data']]
    gt_IN = _gt[my_test_newnew.get_data_name(data_name)['gt']]
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

    data_zeropadding = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)),
                                  'constant', constant_values=0)
        
    whole_indices = sampling_all(gt)
    
    all_data = np.zeros((len(whole_indices), 2*patch_length + 1, 2*patch_length + 1, data_IN.shape[2]))
    
    all_assign = my_test_newnew.indexToAssignment(whole_indices, data.shape[0], data.shape[1], patch_length)
    for i in range(len(all_assign)):
        all_data[i] = my_test_newnew.selectNeighboringPatch(data_zeropadding, all_assign[i][0], all_assign[i][1], patch_length)
        
    all_data = np.expand_dims(all_data, axis=4)

    print('loading model')
 
    model = load_model('%s/resnet_3D_%depoch_%s_%d.h5' % (my_dir, epochs, data_name, index_iter))
    if best_weights_path != None:
        try:
            model.load_weights(best_weights_path)
        except IOError:
            print('There is no saved best model')
    
    print('predicting in all')
    pred = model.predict(all_data).argmax(axis=1)
    #x = np.ravel(pred_test)
    x = pred
    
    print('painting')
    #y = np.zeros((x.shape[0], 3))
    y_re = np.zeros(shape = (data_IN.shape[0], data_IN.shape[1], 3))
    
    for index, item in enumerate(x):
        [row, col] = indexToAssignment_raw(whole_indices[index],data_IN.shape[1])
        if item == 0:
            y_re[row][col] = np.array([192, 192, 0]) / 255.
        if item == 1:
            y_re[row][col] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y_re[row][col] = np.array([0, 255, 0]) / 255.
        if item == 3:
            y_re[row][col] = np.array([0, 0, 255]) / 255.
        if item == 4:
            y_re[row][col] = np.array([255, 255, 0]) / 255.
        if item == 5:
            y_re[row][col] = np.array([0, 255, 255]) / 255.
        if item == 6:
            y_re[row][col] = np.array([255, 0, 255]) / 255.
        if item == 7:
            y_re[row][col] = np.array([192, 192, 192]) / 255.
        if item == 8:
            y_re[row][col] = np.array([128, 128, 128]) / 255.
        if item == 9:
            y_re[row][col] = np.array([128, 0, 0]) / 255.
        if item == 10:
            y_re[row][col] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y_re[row][col] = np.array([0, 0, 128]) / 255.
        if item == 12:
            y_re[row][col] = np.array([128, 128, 0]) / 255.
        if item == 13:
            y_re[row][col] = np.array([128, 0, 128]) / 255.
        if item == 14:
            y_re[row][col] = np.array([0, 128, 128]) / 255.
        if item == 15:
            y_re[row][col] = np.array([192, 0, 192]) / 255.
        if item == 16:
            y_re[row][col] = np.array([0, 0, 192]) / 255.
                
    
    #y_re = np.reshape(y, (gt_IN.shape[0], gt_IN.shape[1], 3))
    
    classification_map(y_re, gt_IN, 24,
                       '%s/res3D_%depoch_%s_%d.png' % (my_dir, epochs, data_name, index_iter))