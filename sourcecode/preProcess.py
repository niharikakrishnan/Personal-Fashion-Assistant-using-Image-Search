##
## Script Name: preProcess.py
## Description: Script to download images and preprocess
##

# 1. Import necessary modules
import pandas as pd
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import multiprocessing as mp

from keras.preprocessing import image
from PIL import Image

# 2. Function to plot images
def show_imgs(imgs, n_row = 3, n_col = 2):
    '''
    This method plots images.
    '''
    try:
        _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
        axs = axs.flatten()
        for img, ax in zip(imgs, axs):
            ax.imshow(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
        plt.show()
    except Exception as e:
        raise Exception('Failure in show_imgs {}'.format(str(e)))
    
# 3. Function to count faces in an image
def face_count(img_nm):
    '''
    This method is used to count number of faces in an image 
    using haarcascade.
    '''
    try:
        # Read the image
        img = cv2.imread(img_nm)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        return (len(faces))
    except Exception as e:
        raise Exception('Failure in face_count {}'.format(str(e)))

# 4. Function to remove background
def remove_background(img, threshold=250.0):
    """
    This method removes background from an image.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        cnts = cv2.findContours(morphed, 
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        mask = cv2.drawContours(threshed, cnt, 0, (0, 255, 0), 0)
        masked_data = cv2.bitwise_and(img, img, mask=mask)

        x, y, w, h = cv2.boundingRect(cnt)
        dst = masked_data[y: y + h, x: x + w]

        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(dst)

        rgba = [r, g, b, alpha]
        dst = cv2.merge(rgba, 4)

        return dst
    except Exception as e:
        raise Exception('Failure in remove_background {}'.format(str(e)))

# 5. Function to resize a given image
def resize(image_nm , newSize = (456,456) ):
    '''
    This method resizes image to specified size using INTER_AREA interpolation.
    '''
    try:
        image_arr = cv2.imread(image_nm)
        resized_img = cv2.resize(image_arr, newSize, interpolation=cv2.INTER_AREA)
        return resized_img
    except Exception as e:
        raise Exception('Failure in resize {}'.format(str(e)))

# 6. Function to save a processed image
def save_img(img_arr, location):
    '''
    This method saves a image to destination location.
    '''
    try:
        cv2.imwrite(location, img_arr)
    except Exception as e:
        raise Exception('Failure in save_img {}'.format(str(e)))

# 7. Function to extract apparel from an image
def get_dress(img_nm,stack=False):
    '''
    This method extracts apparel from an image. 
    This is limited to top wear and full body dresses (wild and studio working). 
    This takes input as rgb and return PNG.
    '''
    try:
        img_arr = cv2.imread(img_nm)
        img_arr = tf.image.resize_with_pad(img_arr,target_height=512,target_width=512)
        rgb  = img_arr.numpy()
        img_arr = np.expand_dims(img_arr,axis=0)/ 255.
        seq = topWear.predict(img_arr)
        seq = seq[3][0,:,:,0]
        seq = np.expand_dims(seq,axis=-1)
        c1x = rgb*seq
        c2x = rgb*(1-seq)
        cfx = c1x+c2x
        dummy = np.ones((rgb.shape[0],rgb.shape[1],1))
        rgbx = np.concatenate((rgb,dummy*255),axis=-1)
        rgbs = np.concatenate((cfx,seq*255.),axis=-1)
        if stack:
            stacked = np.hstack((rgbx,rgbs))
            return stacked
        else:
            return rgbs
    except Exception as e:
        raise Exception('Failure in get_dress {}'.format(str(e)))

# 8. Flip a given image
def flip_image(image_path):
    """
    Flip or mirror the image
    """
    try:
        image_obj = Image.open(image_path)
        rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        return rotated_image
    except Exception as e:
        raise Exception('Failure in flip_image {}'.format(str(e)))