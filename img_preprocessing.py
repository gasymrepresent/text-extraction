#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 maro <maro@mar>
#
# Distributed under terms of the MIT license.

"""
This a main Python programm for the image pre-processing.
"""


# load all  the necessary packages
import os
import cv2
#import numpy as np
#import imutils
from imtools import get_imlist, variance_of_laplacian, increase_brightness, contrast_enh

# useful parameters 
p = 300.0
b = 60

# use the function 'get_imlist'  to call the images' path
imlist = get_imlist('data/demo')

# process the denoising for all the images in 'imlist'
for im in imlist:
    # Load the image
    img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
    orig = img.copy()
    #compute the focus measure of the image using the Variance of Laplacian
    # method
    fm = variance_of_laplacian(img)
    
    # if the focus measure is greater than the supplied threshold,
    # then the image should be considered "Not blurry" else "Blurry" 
    if  fm > p:
        text = "Not blurry"
        print('[INFO] Image already good!')
        # show the image
        #cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        #cv2.imshow("Not blurry image", img)
        # save the results
        enh_img = img
        cv2.imwrite('data/results/tmp.jpg', enh_img)
        
    else:
        text = "Blurry"
        # show the image
        #cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        #cv2.imshow("Blurry image", img)
        print('[INFO] Processing the image...')
        #brightened = increase_brightness(img, b)
        #contrasted = contrast_enh(brightened)
        blur= cv2.medianBlur(img, 5)
        enh_img = cv2.fastNlMeansDenoisingColored(img, None,3, 121, 7, 21)
        # save the results
        filename = "{}.jpg".format(os.getpid())
        cv2.imwrite('data/results/{}'.format(filename), enh_img)
    cv2.imshow('Enhanced image', enh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


