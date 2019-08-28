#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:38:23 2018

@author: jiang
"""

import numpy as np
from keras.models import load_model
from operator import truediv

import collections
from sklearn import metrics

def AA_andEachClassAccuracy(confusion_matrix):
#    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# =============================================================================
# def outputStats(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, loss_and_metrics, CATEGORY, path, path2):
# 
# 
#     f = open(path, 'a')
# 
#     sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
#     f.write(sentence0)
#     sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
#     f.write(sentence1)
#     sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
#     f.write(sentence2)
# 
#     element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
#     element_std = np.std(ELEMENT_ACC_AE, axis=0)
#     sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
#     f.write(sentence5)
#     sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
#     f.write(sentence6)
# 
#     print('Test score:', loss_and_metrics[0])
#     sentence7 = 'Test score:' + str(loss_and_metrics[0]) + '\n'
#     f.write(sentence7)
#     print('Test accuracy:', loss_and_metrics[1])
#     sentence8 = 'Test accuracy:' + str(loss_and_metrics[1]) + '\n'
#     f.write(sentence8)
#     
#     f.close()
# 
# # =============================================================================
# #     print_matrix = np.zeros((CATEGORY), dtype=object)
# #     for i in range(CATEGORY):
# #         print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])
# # =============================================================================
#     print_matrix = np.zeros((CATEGORY), dtype=object)
#     for i in range(CATEGORY):
#         print_matrix[i] = str(ELEMENT_ACC_AE[i])
# 
#     np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
#                newline='\n')
# =============================================================================

def outputStats(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TRAINING_TIME_AE, TESTING_TIME_AE, loss_and_metrics, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:\n' + str(KAPPA_AE) +'\n'+ str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:\n' + str(OA_AE) +'\n'+ str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:\n' + str(AA_AE) +'\n'+ str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n\n'
    f.write(sentence2)
    sentence3 = 'Total average Training time is :\n' + str(TRAINING_TIME_AE) +'\n'+ str(np.sum(TRAINING_TIME_AE)) + '\n\n'
    f.write(sentence3)
    sentence4 = 'Total average Testing time is:\n' + str(TESTING_TIME_AE) +'\n'+ str(np.sum(TESTING_TIME_AE)) + '\n\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = 'Mean of all elements in confusion matrix:\n' + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n\n'
    f.write(sentence5)
    sentence6 = 'Standard deviation of all elements in confusion matrix\n' + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n\n'
    f.write(sentence6)
    
    sentence7 = 'Loss and Accuracy of models are:\n' + str(loss_and_metrics) + '\n\n'
    f.write(sentence7)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str),
               fmt='%s', delimiter="\t", newline='\n')

#    np.savetxt(path2, loss_and_metrics.astype(str), fmt='%s', delimiter="\t",
#               newline='\n')


# =============================================================================
# def Records_output(model, test_data, y_test, KAPPA_RES, OA_RES, AA_RES, loss_and_metrics, CATEGORY,
#                    TRAINING_TIME, TESTING_TIME):
#     print('shape_test_data:', test_data.shape, 'shape_y_test:', y_test.shape)
#     
#     print('predicting model')
#     pred_test = model.predict(test_data).argmax(axis=1)
#     collections.Counter(pred_test)
#     #gt_test = gt[test_indices] - 1
#     overall_acc = metrics.accuracy_score(pred_test, y_test.argmax(axis=1))
#     confusion_matrix = metrics.confusion_matrix(pred_test, y_test.argmax(axis=1))
#     each_acc, average_acc = AA_andEachClassAccuracy(confusion_matrix)
#     kappa = metrics.cohen_kappa_score(pred_test, y_test.argmax(axis=1))
#     ELEMENT_ACC = each_acc
#     KAPPA_RES.append(kappa)
#     OA_RES.append(overall_acc)
#     AA_RES.append(average_acc)
#     
#     outputStats(KAPPA_RES, OA_RES, AA_RES, ELEMENT_ACC, loss_and_metrics, CATEGORY = CATEGORY,
#                 path = '/home/jiang/SSRN-master/records/UP_records.txt',
#                 path2 = '/home/jiang/SSRN-master/records/UP_records_element.txt')
# =============================================================================
