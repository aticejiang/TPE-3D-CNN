# Implementation of TPE-3D-CNN for hyperspectral image classification

## Descriptions

A deep residual 3D convolutional neural network (TPE-3D-CNN) framework is proposed for hyperspectral images classification in order to realize fast training, classification and parameter optimization. It takes raw 3D cubes as input data without feature engineering. And the TPE algorithm is introduced to optimize hyperparameters adaptively.

![Fig.1 3D-CNN optimized manuelly](https://github.com/aticejiang/TPE_3D_CNN/raw/master/figure/figure3.png)

*Fig.1 3D-CNN optimized manuelly*

![Fig.2 3D-CNN optimized by TPE algorithm](https://github.com/aticejiang/TPE_3D_CNN/raw/master/figure/figure5.png)

*Fig.2 3D-CNN optimized by TPE algorithm*

![Fig.3 TPE Search Space](https://github.com/aticejiang/TPE_3D_CNN/raw/master/figure/TPE_Searchspace.png)

*Fig.3 TPE search space setting: the TPE algorithm works in this space in order to have the best perform of networks, in the limited number of iterations.*


The hyperparameters in manuel networks proposed is pre-setting by manuel searchs. The parts inside the dotted frame are included in hyperparameter search space for TPE algorithm. As result, the network optimized by TPE algorithm reduces the number of trainable parameters by half and the training time by about 10%, compared with manual settings. And the accuracy (OA, AA, Kappa) has also been promoted.

## Prerequisites
* [Anaconda 3](https://www.anaconda.com/distribution/)
* [Tensorflow 1.4](https://pypi.org/project/tensorflow-gpu/1.4.0/)
* [Keras 2.0](https://pypi.org/project/Keras/)

## Usage
Train models with commonly studied hyperspectral imagery (HSI) datasets:

run TPE-3D-CNN: 
`<$ python ./TPE/hyperas_my_test_run.py>` 

run manuel-3D-CNN
`<$ python ./manuel/my_test_newnewnewnew.py>` 

You may edit the files to train different iteration, epochs and datasets. If you encountered problems like " no such file or direcotry", please check the corresponding paths and change them to absolute paths.

