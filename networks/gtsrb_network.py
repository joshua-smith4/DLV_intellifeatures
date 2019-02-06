#!/usr/bin/env python

from __future__ import print_function

import scipy.io as sio
import numpy as np
import copy
import os
import csv
import cv2
from sklearn.utils import shuffle

from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

from keras.optimizers import SGD

batch_size = 32
nb_classes = 43
nb_epoch = 200
# data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3


def load_gtsrb_training_data(data_path):
    ''' data_path - path do directory containing folders of all
    '''
    imgs = []
    labels = []
    for i in range(nb_classes):
        direct_name = str(i).zfill(5)
        direct_path = os.path.join(data_path, direct_name)
        stat_file = os.path.join(direct_path, 'GT-' + direct_name + '.csv')
        assert(os.path.exists(stat_file))
        with open(stat_file) as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                file_path = os.path.join(direct_path, row[0])
                if not file_path.endswith('.ppm'):
                    continue
                img = cv2.imread(file_path)
                # hsv = color.rgb2hsv(img)
                # hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
                # img = color.hsv2rgb(hsv)
                # min_side = min(img.shape[:-1])
                # centre = img.shape[0]//2, img.shape[1]//2
                # img = img[centre[0]-min_side//2:centre[0]+min_side//2,
                #         centre[1]-min_side//2:centre[1]+min_side//2,
                #         :]
                img = cv2.resize(img, (img_rows, img_cols))
                img = np.rollaxis(img, -1)
                imgs += [img]
                labels += [i]
    num_samples = len(imgs)
    boundary = int(num_samples*0.8)
    imgs, labels = shuffle(imgs, labels, random_state=0)
    X_train = np.array(imgs[0:boundary], dtype=np.float32) / 255.0
    X_test = np.array(imgs[boundary:], dtype=np.float32) / 255.0
    Y_train = np_utils.to_categorical(labels[0:boundary], nb_classes)
    Y_test = np_utils.to_categorical(labels[boundary:], nb_classes)
    return X_train, Y_train, X_test, Y_test

dataset_location = 'networks/GTSRB'

def read_dataset():

    (X_train, Y_train, X_test, Y_test) = load_gtsrb_training_data(
        os.path.join(dataset_location, 'Final_Training', 'Images'))
    print('X_train shape:', X_train.shape)
    print('Y_train length:', len(Y_train))
    print('X_test shape:', X_test.shape)
    print('Y_test length:', len(Y_test))
    # print(X_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices
    return (X_train, Y_train, X_test, Y_test, img_channels, img_rows, img_cols, batch_size, nb_classes, nb_epoch)


def build_model(img_channels, img_rows, img_cols, nb_classes):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, img_rows, img_cols), activation='relu'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def read_model_from_file(img_channels, img_rows, img_cols, nb_classes, weightFile, modelFile):
    """
    define neural network model
    :return: network model
    """
    weights = sio.loadmat(weightFile)
    model = model_from_json(open(modelFile).read())
    model.summary()

    for (idx,lvl) in [(1,0),(2,1),(3,4),(4,5),(5,8),(6,9),(7,13),(8,15)]:
        weight_1 = 2 * idx - 2
        weight_2 = 2 * idx - 1
        model.layers[lvl].set_weights([weights['weights'][0, weight_1], weights['weights'][0, weight_2].flatten()])

    return model


"""
   The following function gets the activations for a particular layer
   for an image in the test set.
   FIXME: ideally I would like to be able to
          get activations for a particular layer from the inputs of another layer.
"""

(_, _,X_test_verif,Y_test_verif,_,_,_,_,_,_) = read_dataset()

def getImage(model, n_in_tests):
    return X_test_verif[n_in_tests]


def readImage(path):
    im = cv2.resize(cv2.imread(path), (img_rows, img_cols)).astype('float32')
    im /= 255.0
    im = np.rollaxis(im, -1)

    return np.squeeze(im)


def getActivationValue(model, layer, image):

    image = np.expand_dims(image, axis=0)
    activations = get_activations(model, layer, image)
    return np.squeeze(activations)


def get_activations(model, layer, X_batch):
    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch, 0])
    return activations


def predictWithImage(model, newInput):

    newInput_for_predict = copy.deepcopy(newInput)
    newInput2 = np.expand_dims(newInput_for_predict, axis=0)
    predictValue = model.predict(newInput2)
    newClass = np.argmax(np.ravel(predictValue))
    confident = np.amax(np.ravel(predictValue))
    return (newClass, confident)


def getWeightVector(model, layer2Consider):
    weightVector = []
    biasVector = []

    for layer in model.layers:
    	 index=model.layers.index(layer)
         h=layer.get_weights()

         if len(h) > 0 and index in [0,1,4,5,8,9]  and index <= layer2Consider:
         # for convolutional layer
             ws = h[0]
             bs = h[1]

             #print("layer =" + str(index))
             #print(layer.input_shape)
             #print(ws.shape)
             #print(bs.shape)

             # number of filters in the previous layer
             m = len(ws)
             # number of features in the previous layer
             # every feature is represented as a matrix
             n = len(ws[0])

             for i in range(1,m+1):
                 biasVector.append((index,i,h[1][i-1]))

             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1):
                     # (feature, filter, matrix)
                     weightVector.append(((index,j),(index,i),v[j-1]))

         elif len(h) > 0 and index in [13,15]  and index <= layer2Consider:
         # for fully-connected layer
             ws = h[0]
             bs = h[1]

             # number of nodes in the previous layer
             m = len(ws)
             # number of nodes in the current layer
             n = len(ws[0])

             for j in range(1,n+1):
                 biasVector.append((index,j,h[1][j-1]))

             for i in range(1,m+1):
                 v = ws[i-1]
                 for j in range(1,n+1):
                     weightVector.append(((index-1,i),(index,j),v[j-1]))
         #else: print "\n"

    return (weightVector, biasVector)


def getConfig(model):

    config = model.get_config()
    config = [getLayerName(dict) for dict in config]
    config = zip(range(len(config)), config)
    return config


def getLayerName(dict):

    className = dict.get('class_name')
    if className == 'Activation':
        return dict.get('config').get('activation')
    else:
        return className
