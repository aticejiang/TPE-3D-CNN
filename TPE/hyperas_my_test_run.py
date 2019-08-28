#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 00:26:58 2018

@author: jiang

improvement: 
    1.Tensorboard
    2.recoards
    3.multi-training
    4.ReduceLROnPlateau
    5.model layers reduction 31 -> 19
    6.best_weights evaluation records
    7.hyperas hyper-parameters
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
import hyperas_domain_data
import hyperas_my_test_model
import hyperas_my_test_draw
import my_test_newnew_record
from keras.utils import plot_model
import pickle
from keras.models import load_model

import collections
from sklearn import metrics, preprocessing

#times of model training 
ITER = 1

batch_size = 16
epochs = 2
patch_length = 3
data_name = 'KSC'

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
    
image = '/home/jiang/SSRN-master/datasets/%s/%s.mat' % (directory, hyperas_domain_data.get_data_name(data_name)['data'])
ground_truth = '/home/jiang/SSRN-master/datasets/%s/%s.mat' % (directory, hyperas_domain_data.get_data_name(data_name)['gt'])

KAPPA_RES = []
OA_RES = []
AA_RES = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, hyperas_domain_data.get_data_name(data_name)['category']))
LOSS_and_METRICS = np.zeros((ITER, 2))

KAPPA_RES_best = []
OA_RES_best = []
AA_RES_best = []
ELEMENT_ACC_best = np.zeros((ITER, hyperas_domain_data.get_data_name(data_name)['category']))
LOSS_and_METRICS_best = np.zeros((ITER, 2))

seeds = [1222, 1223, 1222, 1223, 1224, 1223, 1226, 1227, 1228, 1229]
#seeds = [1440, 1127, 1440, 1440, 1127, 1223, 1333, 1228, 1229]

my_dir = '/home/jiang/SSRN-master/upload_test'

for index_iter in xrange(ITER):
    print("# %d Iteration" % (index_iter + 1))
          
    path_dir = '%s/models' % (my_dir)
    isExists=os.path.exists(path_dir)
    if not isExists:
        os.makedirs(path_dir)
        print path_dir+' Successfully Maked! '
    best_weights_path = '%s/models/resnet_3d_%depoch_%s_%d.hdf5' % (my_dir, epochs, data_name, index_iter + 1)
    
    np.random.seed(seeds[index_iter])
    
    train_data, y_train, validation_data, y_validation, test_data, y_test = hyperas_domain_data.data_preprocessing(image, ground_truth,
                                                                                                           patch_length = patch_length,
                                                                                                           training_rate = 0.2,
                                                                                                           validation_rate = 0.1,
                                                                                                           data_name = data_name)
    
    
    model = hyperas_my_test_model.ResNet50_3D(weights=None,input_tensor=None, 
                                              input_shape=train_data.shape[1:], classes=y_train.shape[1])
    
#    model.load_weights('/home/jiang/SSRN-master/data_aug/TPE/PaviaUniversity/pure_vs_aug/aug/2/models/resnet_3d_150epoch_PaviaU_1.hdf5',by_name=True)
#    for layer in model.layers[:-5]:
#        layer.trainable = False
    
    adamp = Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=adamp, metrics=['accuracy'])
#    model = load_model('/home/jiang/SSRN-master/data_aug/PaviaUniversity/2adamp/resnet_3D_150epoch_PaviaU_1.h5')
#    model.load_weights(best_weights_path)
#    K.set_value(model.optimizer.lr, 0.2 * K.get_value(model.optimizer.lr))
#    print('lr = ',K.get_value(model.optimizer.lr))
    
    earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    saveBestModel = kcallbacks.ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')
    tensorboard_log = '%s/records/tensorboard/%s%d/%d' % (my_dir, data_name, epochs, index_iter + 1)
    tensorBoard = kcallbacks.TensorBoard(log_dir= tensorboard_log)
    reduceLROnPlateau = kcallbacks.ReduceLROnPlateau(patience = 15)
#    Tnan = kcallbacks.TerminateOnNaN()
    
    if index_iter == 0:
        plot_model(model, to_file = '%s/resnet_3d_model.png' % (my_dir))
        model.summary()
        
    tic6 = time.clock()
    history = model.fit(train_data, y_train,
                        validation_data=(validation_data, y_validation),
                        batch_size=batch_size,
                        epochs=epochs, shuffle=True, callbacks=[earlyStopping, saveBestModel, tensorBoard, reduceLROnPlateau],
                        initial_epoch = 0)
    toc6 = time.clock()
    
    with open('%s/records/trainHistoryDict_%s_%d' % (my_dir, data_name, index_iter + 1), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
#    with open('/home/jiang/SSRN-master/records/trainHistoryDict_%s_%d' % (data_name, index_iter + 1), 'rb') as file_pi:  
#        load_history=pickle.load(file_pi) 
        
    model.save('%s/resnet_3D_%depoch_%s_%d.h5' % (my_dir, epochs, data_name, index_iter + 1))
    
        
    tic7 = time.clock()
    loss_and_metrics = model.evaluate(test_data, y_test, batch_size = batch_size)
    print('loss:', loss_and_metrics[0])
    print('accuracy:', loss_and_metrics[1])
    toc7 = time.clock()
    
    print('# %d Iteration Training Time: ' % (index_iter + 1), toc6 - tic6)
    print('# %d Iteration Test time:' %(index_iter + 1), toc7 - tic7)
    print(history.history.keys())
    
##    draw map
    hyperas_my_test_draw.classification_draw(my_dir, image, ground_truth, patch_length, data_name, 
                                             epochs, index_iter + 1, best_weights_path = best_weights_path)
    
#    records
    print('predicting model in test')
    pred_test = model.predict(test_data).argmax(axis=1)
    
    collections.Counter(pred_test)
    overall_acc = metrics.accuracy_score(pred_test, y_test.argmax(axis=1))
    confusion_matrix = metrics.confusion_matrix(pred_test, y_test.argmax(axis=1))
    each_acc, average_acc = my_test_newnew_record.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, y_test.argmax(axis=1))
    KAPPA_RES.append(kappa)
    OA_RES.append(overall_acc)
    AA_RES.append(average_acc)
    TRAINING_TIME.append(toc6 - tic6)
    TESTING_TIME.append(toc7 - tic7)
    ELEMENT_ACC[index_iter, :] = each_acc
    LOSS_and_METRICS[index_iter, :] = loss_and_metrics
    
#    best records
    model.load_weights(best_weights_path)
    loss_and_metrics = model.evaluate(test_data, y_test, batch_size = batch_size)
    print('predicting model in test')
    pred_test = model.predict(test_data).argmax(axis=1)
    collections.Counter(pred_test)
    overall_acc = metrics.accuracy_score(pred_test, y_test.argmax(axis=1))
    confusion_matrix = metrics.confusion_matrix(pred_test, y_test.argmax(axis=1))
    each_acc, average_acc = my_test_newnew_record.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, y_test.argmax(axis=1))
    KAPPA_RES_best.append(kappa)
    OA_RES_best.append(overall_acc)
    AA_RES_best.append(average_acc)
    ELEMENT_ACC_best[index_iter, :] = each_acc
    LOSS_and_METRICS_best[index_iter, :] = loss_and_metrics
    
    print('# %d Iteration Training Finished.' % (index_iter + 1) )
    
my_test_newnew_record.outputStats(KAPPA_RES, OA_RES, AA_RES, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                                  LOSS_and_METRICS, CATEGORY = hyperas_domain_data.get_data_name(data_name)['category'],
                                  path1 = '%s/records/%s_records.txt' % (my_dir, directory),
                                  path2 = '%s/records/%s_records_element.txt' % (my_dir, directory))

my_test_newnew_record.outputStats(KAPPA_RES_best, OA_RES_best, AA_RES_best, ELEMENT_ACC_best, TRAINING_TIME, TESTING_TIME,
                                  LOSS_and_METRICS_best, CATEGORY = hyperas_domain_data.get_data_name(data_name)['category'],
                                  path1 = '%s/records/%s_best_records.txt' % (my_dir, directory),
                                  path2 = '%s/records/%s_best_records_element.txt' % (my_dir, directory))