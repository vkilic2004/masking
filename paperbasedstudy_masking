# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:41:33 2019

@author: volkan
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# read gray image generated in matlab
im = cv.imread('closeBW.bmp')
plt.figure(0)
plt.imshow(im)

# convert gray image to single channel image
imgray=im[:,:,2]
imgray = np.array(imgray)
plt.figure(1)
plt.imshow(imgray)

# convert it to binary
imgray[imgray >= 255] = 1


# read rgb image
im_rgb = cv.imread('circularimage.bmp')
plt.figure(2)
plt.imshow(im_rgb)


# convert single channel imgray image to three channel for masking
imgray3 = np.zeros_like(im_rgb)
imgray3[:,:,0] = imgray
imgray3[:,:,1] = imgray
imgray3[:,:,2] = imgray


# mask rgb image
croppedCircularImage= cv.multiply(im_rgb,imgray3)
plt.figure(3)
plt.imshow(croppedCircularImage)

# calculate average R channel value
pixelpointsCV2 = cv.findNonZero(imgray)
numberofnonzeroelements= pixelpointsCV2.shape[0]

sumvalues=cv.sumElems(croppedCircularImage[:,:,2])[0]

average=sumvalues/numberofnonzeroelements
