#!/usr/bin/env python

"""
compupute e_k according to e_{k-1} and p_{k-1}
author: Xiaowei Huang
"""

import numpy as np
import copy
from scipy import ndimage

from configuration import *
from imageNet_network import addZeroPadding2D
from networkBasics import *


############################################################
#
#  initialise a region for the input
#
################################################################


def initialiseRegionActivation(model,manipulated,image):

    config = NN.getConfig(model)

    # get the type of the current layer
    layerType = getLayerType(model,0)
    #[ lt for (l,lt) in config if l == 0 ]
    #if len(layerType) > 0: layerType = layerType[0]
    #else: print "cannot find the layerType"

    if layerType == "Convolution2D":
        nextSpan = {}
        nextNumSpan = {}
        if len(image.shape) == 2:
            # decide how many elements in the input will be considered
            if len(image)*len(image[0])  < featureDims :
                numDimsToMani = len(image)*len(image[0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop2DActivation(image,manipulated,[],numDimsToMani,-1)

        elif len(image.shape) == 3:
            # decide how many elements in the input will be considered
            if len(image)*len(image[0])*len(image[0][0])  < featureDims :
                numDimsToMani = len(image)*len(image[0])*len(image[0][0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop3DActivation(image,manipulated,[],numDimsToMani,-1)

        for i in ls:
            nextSpan[i] = span
            nextNumSpan[i] = numSpan

    elif layerType == "InputLayer":
        nextSpan = {}
        nextNumSpan = {}
        # decide how many elements in the input will be considered
        if len(image)  < featureDims :
            numDimsToMani = len(image)
        else: numDimsToMani = featureDims
        # get those elements with maximal/minimum values
        ls = getTopActivation(image,manipulated,-1,numDimsToMani)
        for i in ls:
            nextSpan[i] = span
            nextNumSpan[i] = numSpan

    elif layerType == "ZeroPadding2D":
        #image1 = addZeroPadding2D(image)
        image1 = image
        nextSpan = {}
        nextNumSpan = {}
        if len(image1.shape) == 2:
            # decide how many elements in the input will be considered
            if len(image1)*len(image1[0])  < featureDims :
                numDimsToMani = len(image1)*len(image1[0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop2DActivation(image1,manipulated,[],numDimsToMani,-1)

        elif len(image1.shape) == 3:
            # decide how many elements in the input will be considered
            if len(image1)*len(image1[0])*len(image1[0][0])  < featureDims :
                numDimsToMani = len(image1)*len(image1[0])*len(image1[0][0])
            else: numDimsToMani = featureDims
            # get those elements with maximal/minimum values
            ls = getTop3DActivation(image1,manipulated,[],numDimsToMani,-1)
        for i in ls:
            nextSpan[i] = span
            nextNumSpan[i] = numSpan

    else:
        print "initialiseRegionActivation: Unknown layer type ... "

    return (nextSpan,nextNumSpan,numDimsToMani)




############################################################
#
#  auxiliary functions
#
################################################################

# This function only suitable for the input as a list, not a multi-dimensional array

def getTopActivation(image,manipulated,layerToConsider,numDimsToMani):
    avoid = repeatedManipulation == "disallowed"
    diffs = np.zeros((len(averages[layerToConsider+1]),) + image.shape, dtype=np.float32)
    distances = np.zeros(diffs.shape[0], dtype=np.float32)
    for i in range(len(distances)):
        diffs[i] = np.abs(image - np.squeeze(averages[layerToConsider+1][i]))
        distances[i] = np.sqrt(np.sum(np.square(diffs[i])))
    closestClassIndex = np.argsort(distances)[1]
    maxDims = np.unravel_index(np.argsort(-1*diffs[closestClassIndex], axis=None), diffs[closestClassIndex].shape)
    convertedMaxDims = [(maxDims[0][i],) for i in range(len(maxDims[0]))]
    finalList = []
    for i in range(len(convertedMaxDims)):
        if (not avoid) or (convertedMaxDims[i] in manipulated):
            continue
        finalList.append(convertedMaxDims[i])
    return finalList[:numDimsToMani]

def getTop2DActivation(image,manipulated,ps,numDimsToMani,layerToConsider):
    avoid = repeatedManipulation == "disallowed"
    # avg = np.sum(image)/float(len(image)*len(image[0]))
    diffs = np.zeros((len(averages[layerToConsider+1]),) + image.shape, dtype=np.float32)
    distances = np.zeros(diffs.shape[0], dtype=np.float32)
    for i in range(len(distances)):
        diffs[i] = np.abs(image - np.squeeze(averages[layerToConsider+1][i]))
        distances[i] = np.sqrt(np.sum(np.square(diffs[i])))
    closestClassIndex = np.argsort(distances)[1]
    maxDims = np.unravel_index(np.argsort(-1*diffs[closestClassIndex], axis=None), diffs[closestClassIndex].shape)
    convertedMaxDims = [(maxDims[0][i], maxDims[1][i]) for i in range(len(maxDims[0]))]
    finalList = []
    for i in range(len(convertedMaxDims)):
        if (not avoid) or (convertedMaxDims[i] in manipulated):
            continue
        finalList.append(convertedMaxDims[i])
    return finalList[:numDimsToMani]
# original

# ps are indices of the previous layer


def getTop3DActivation(image,manipulated,ps,numDimsToMani,layerToConsider):
    # print('ps: {}'.format(ps))
    avoid = repeatedManipulation == "disallowed"
    # print('image shape: {}'.format(image.shape))
    # print('average shape: {}'.format(np.squeeze(averages[layerToConsider+1][0]).shape))
    diffs = np.zeros((len(averages[layerToConsider+1]),) + image.shape, dtype=np.float32)
    distances = np.zeros(diffs.shape[0], dtype=np.float32)
    for i in range(len(distances)):
        diffs[i] = np.abs(image - np.squeeze(averages[layerToConsider+1][i]))
        distances[i] = np.sqrt(np.sum(np.square(diffs[i])))
    closestClassIndex = np.argsort(distances)[1]
    maxDims = np.unravel_index(np.argsort(-1*diffs[closestClassIndex], axis=None), diffs[closestClassIndex].shape)
    convertedMaxDims = [(maxDims[0][i], maxDims[1][i], maxDims[2][i]) for i in range(len(maxDims[0]))]
    res = []
    if len(ps) > 0:
        if len(ps[0]) == 3:
            (p1,p2,p3) = zip(*ps)
            ps = zip(p2,p3)
    pointsToConsider = []
    for j in range(numDimsToMani):
        nextPoint = []
        if j <= len(ps) - 1:
            (x,y) = ps[j]
            nps = [ (x-x1,y-y1) for x1 in range(filterSize) for y1 in range(filterSize) if x-x1 >= 0 and y-y1 >=0 ]
            pointsToConsider += nps
            nextPoint = myFindFromArea3D(image, manipulated, avoid, convertedMaxDims, nps, 1, res)
        else:
            nextPoint = myFindFromArea3D(image, manipulated, avoid, convertedMaxDims, pointsToConsider, 1, res)
        res += [nextPoint]
    # print(res)
    return res

def myFindFromArea3D(image, manipulated, avoid, nimage, ps, numDimsToMani, ks):
    for i in range(len(nimage)):
        (x,y,z) = nimage[i]
        if ((y,z) in ps or len(ps) == 0) and ((x,y,z) not in ks) and ((x,y,z) not in manipulated):
            print('selected index: {}'.format(i))
            return nimage[i]


def findFromArea3D(image,manipulated,avoid,nimage,ps,numDimsToMani,ks):
    topImage = {}
    toBeDeleted = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            for k in range(len(image[0][0])):
                if len(topImage) < numDimsToMani and ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks:
                    topImage[(i,j,k)] = nimage[i][j][k]
                elif ((j,k) in ps or len(ps) == 0) and (i,j,k) not in ks:
                    bl = False
                    for (k1,k2,k3), v in topImage.iteritems():
                        if v < nimage[i][j][k] and not ((k1,k2,k3) in toBeDeleted) and ((not avoid) or ((i,j,k) not in manipulated)):
                            toBeDeleted.append((k1,k2,k3))
                            bl = True
                            break
                    if bl == True:
                        topImage[(i,j,k)] = nimage[i][j][k]
    for (k1,k2,k3) in toBeDeleted:
        del topImage[(k1,k2,k3)]
    return topImage.keys()
