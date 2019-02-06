import os, struct
from array import array as pyarray
# from cvxopt.base import matrix
import numpy as np

import PIL.Image

# TODO: find real labels
def LABELS(index):
    return str(index)


def save(layer,image,filename):
    """ Save image in [filename] destination
    """
    import cv2
    import copy

    image_cv = copy.deepcopy(image)
    image_cv = image_cv.transpose(1, 2, 0)

    params = [cv2.IMWRITE_PXM_BINARY, 1]

    cv2.imwrite(filename, image_cv, params)


def show(image):
    """
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    #image = image.reshape(3,32,32).transpose(1,2,0)
    imgplot = ax.imshow(image.T, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
