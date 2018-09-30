#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 maro <maro@mar>
#
# Distributed under terms of the MIT license.

"""
This is a Python programm to define some tools
that we are going to use during the image pre-processing.
"""

# load the necessary packages
import os
import cv2
import numpy as np


# function to retrieve the images path
def get_imlist(path):
    """
    Return a list of filenames for
    all jpg, jpeg, png images in directory
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]


# create 'increase_brightness' to brighten some dark images
def increase_brightness(img, br_val=30):
    """ Brighten the image dependent of the value of br_val """

    # converting image to HSV color model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # splitting the HSV image to different channels
    h, s, v = cv2.split(hsv)

    # increase the value of v-channel if it is inferior
    # to the tunned value
    lim = 255 - br_val
    v[v > lim] = 255
    v[v <= lim] += br_val

    # merge the adjusted v-channel with the h and s channel
    vimg = cv2.merge((h, s, v))

    # converting image from HSV color model to RGB model
    brightened = cv2.cvtColor(vimg, cv2.COLOR_HSV2BGR)
    return brightened


def contrast(img):
    """
    Contrast the image .
    """

    # converting image to lab color model
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # splitting the lab image to different channels
    l, a, b = cv2.split(lab)

    # applying the CHLAE to l-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # merge the CHLAE enhanced l-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    #  converting image from lab color model to RGB model
    contrasted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # return contrasted image
    return contrasted

def contrast_enh(img):

	# convert the image to YUV
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	return img_output

def adjust_gamma(image, gamma=1.0):
    """
    Adjust the gamma effect of an image.
    """

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def contrasting_text_color(hex_str):
	'''
	Input a string without hash sign of RGB hex digits to compute
	complementary contrasting color such as for fonts
	'''

	(r, g, b) = (hex_str[:2], hex_str[2:4], hex_str[4:])
	return '000' if 1 - (int(r, 16) * 0.299 + int(g, 16) * 0.587 + int(b, 16) * 0.114) / 255 < 0.5 else 'fff'

