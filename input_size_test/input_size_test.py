#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:44:01 2018

@author: jiang

improvement: 
    1.Tensorboard
    2.recoards
    3.multi-training
    4.ReduceLROnPlateau
    5.model layers reduction 31 -> 19
    6.best_weights evaluation records
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
from keras import backend as K
import time
import os
import my_test_newnew
import input_size_test_model
import my_test_new_classification
import my_test_newnew_record
from keras.utils import plot_model
import pickle

import collections
from sklearn import metrics, preprocessing

#times of model training 
ITER = 2

batch_size = 16
epochs = 1
patch_lengths = [1, 2, 3, 4, 5]
data_name = 'Indian_Pines'

if data_name == 'Indian_Pines':
    directory = 'IN'
elif data_name == 'PaviaU':
    directory = 'UP'
elif data_name == 'KSC':
    directory = 'KSC'
elif data_name == 'Botswana':
    directory = 'Botswana'
elif data_name == 'PaviaC':
    directory = 'PaviaC'
elif data_name == 'Salinas':
    directory = 'Salinas'
    
image = '/home/jiang/SSRN-master/datasets/%s/%s.mat' % (directory, my_test_newnew.get_data_name(data_name)['data'])
ground_truth = '/home/jiang/SSRN-master/datasets/%s/%s.mat' % (directory, my_test_newnew.get_data_name(data_name)['gt'])


seeds = [1220, 1222, 1223, 1224, 1221, 1223, 1226, 1227, 1228, 1229]

my_dir = '/home/jiang/SSRN-master/upload_test'

for test_iter in range(4,6):
    for index_iter in range(1, ITER):
        print("# %d Iteration" % (index_iter + 1))
              
        path_dir = '%s/%d/models' % (my_dir, test_iter)
        isExists=os.path.exists(path_dir)
        if not isExists:
            os.makedirs(path_dir)
            print path_dir+' Successfully Maked '
        best_weights_path = '%s/%d/models/resnet_3d_%depoch_%s_%d.hdf5' % (my_dir, test_iter, epochs, data_name, index_iter + 1)
        
        np.random.seed(seeds[test_iter])
        patch_length = patch_lengths[test_iter-1]
        
        train_data, y_train, validation_data, y_validation, test_data, y_test = my_test_newnew.data_preprocessing(image, ground_truth,
                                                                                                               patch_length = patch_length,
                                                                                                               validation_rate = 0.1,
                                                                                                               data_name = data_name)
        
        
        model = input_size_test_model.ResNet50_3D(avg_size = 2*patch_length + 1, weights=None,input_tensor=None, 
                                              input_shape=train_data.shape[1:], classes=y_train.shape[1])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        
        earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        saveBestModel = kcallbacks.ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')
        tensorboard_log = '%s/%d/records/tensorboard/%s%d/%d' % (my_dir, test_iter, data_name, epochs, index_iter + 1)
        tensorBoard = kcallbacks.TensorBoard(log_dir= tensorboard_log)
        reduceLROnPlateau = kcallbacks.ReduceLROnPlateau(patience = 15)
        
        if index_iter == 0:
            plot_model(model, to_file = '%s/%d/resnet_3d_model.png' % (my_dir, test_iter))
            model.summary()
            
        tic6 = time.clock()
        history = model.fit(train_data, y_train,
                            validation_data=(validation_data, y_validation),
                            batch_size=batch_size,
                            epochs=epochs, shuffle=True, callbacks=[earlyStopping, saveBestModel, tensorBoard, reduceLROnPlateau])
        toc6 = time.clock()
        
        with open('%s/%d/records/trainHistoryDict_%s_%d' % (my_dir, test_iter, data_name, index_iter + 1), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    #    with open('/home/jiang/SSRN-master/records/trainHistoryDict_%s_%d' % (data_name, index_iter + 1), 'rb') as file_pi:  
    #        load_history=pickle.load(file_pi) 
            
        model.save('%s/%d/resnet_3D_%depoch_%s_%d.h5' % (my_dir, test_iter, epochs, data_name, index_iter + 1))
        
            
        tic7 = time.clock()
        loss_and_metrics = model.evaluate(test_data, y_test, batch_size = batch_size)
        print('loss:', loss_and_metrics[0])
        print('accuracy:', loss_and_metrics[1])
        toc7 = time.clock()
        
        print('# %d Iteration Training Time: ' % (index_iter + 1), toc6 - tic6)
        print('# %d Iteration Test time:' %(index_iter + 1), toc7 - tic7)
        print(history.history.keys())
        
    #    draw map
    #    my_test_new_classification.classification_draw(image, ground_truth, patch_length, data_name, epochs, index_iter + 1)
        
    #    records
    #    print('predicting model in test')
    #    pred_test = model.predict(test_data).argmax(axis=1)
        
    
    #    best records
        try:
            model.load_weights(best_weights_path)
            best_loss_and_metrics = model.evaluate(test_data, y_test, batch_size = batch_size)
        except IOError:
            print('There is no saved best model')
            best_loss_and_metrics = None
        
        trials = {'input size': 2*patch_length + 1,
                  'loss_and_metrics': loss_and_metrics, 'best_loss_and_metrics': best_loss_and_metrics,
                  'Training Time': toc6 - tic6, 'Testing Time': toc7 - tic7}
        with open('%s/%d/records/evaluate_%d.pkl' %(my_dir, test_iter, index_iter + 1), 'wb') as file_pi:
            pickle.dump(trials, file_pi)
        
        print('# %d Iteration Training Finished.' % (index_iter + 1) )


