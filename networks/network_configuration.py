#!/usr/bin/env python

"""
Define paramters
author: Xiaowei Huang
"""
import numpy as np
import os

def network_parameters(dataset):


#######################################################
#
#  boundOfPixelValue for the elements of input
#
#######################################################

    if dataset in ["mnist","cifar10","imageNet","gtsrb"] :
        boundOfPixelValue = [0,1]
    elif dataset == "twoDcurve":
        boundOfPixelValue = [0, 2 * np.pi]
    else:
        boundOfPixelValue = [0,0]

#######################################################
#
#  some auxiliary parameters that are used in several files
#  they can be seen as global parameters for an execution
#
#######################################################



# which dataset to analyse
    if dataset == "mnist":
        import mnist_network as NN_mnist
        import mnist
        NN = NN_mnist
        dataBasics = mnist
        directory_model_string = makedirectory("networks/mnist")
        directory_statistics_string = makedirectory("data/mnist_statistics")
        directory_pic_string = makedirectory("data/mnist_pic")

# ce: the region definition for layer 0, i.e., e_0
        featureDims = 10
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0

    elif dataset == "twoDcurve":
        import twoDcurve_network as NN_twoDcurve
        import twoDcurve
        NN = NN_twoDcurve
        dataBasics = twoDcurve
        directory_model_string = makedirectory("networks/twoDcurve")
        directory_statistics_string = makedirectory("data/twoDcurve_statistics")
        directory_pic_string = makedirectory("data/twoDcurve_pic")

# ce: the region definition for layer 0, i.e., e_0
        featureDims = 2
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0

    elif dataset == "cifar10":
        import cifar10_network as NN_cifar10
        import cifar10
        NN = NN_cifar10
        dataBasics = cifar10
        directory_model_string = makedirectory("networks/cifar10")
        directory_statistics_string = makedirectory("data/cifar10_statistics")
        directory_pic_string = makedirectory("data/cifar10_pic")

# ce: the region definition for layer 0, i.e., e_0
        featureDims = 5
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0

    elif dataset == "imageNet":
        import imageNet_network as NN_imageNet
        import imageNet
        NN = NN_imageNet
        dataBasics = imageNet
        directory_model_string = makedirectory("networks/imageNet")
        directory_statistics_string = makedirectory("data/imageNet_statistics")
        directory_pic_string = makedirectory("data/imageNet_pic")

# ce: the region definition for layer 0, i.e., e_0
        featureDims = 5
        span = 125
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 125

    elif dataset == "gtsrb":
        import gtsrb_network as NN_gtsrb
        import gtsrb
        NN = NN_gtsrb
        dataBasics = gtsrb
        directory_model_string = makedirectory("networks/GTSRB")
        directory_statistics_string = makedirectory("data/gtsrb_statistics")
        directory_pic_string = makedirectory("data/gtsrb_pic")

        featureDims = 5
        span = 255/float(255)
        numSpan = 1
        errorBounds = {}
        errorBounds[-1] = 1.0
#######################################################
#
#  size of the filter used in convolutional layers
#
#######################################################

    filterSize = 3

    return (featureDims,span,numSpan,errorBounds,boundOfPixelValue,NN,dataBasics,directory_model_string,directory_statistics_string,directory_pic_string,filterSize)

def makedirectory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name
