"""Module functions to preprocess images"""
import cv2
import numpy as np


def center_normalize(data):
    """Center normalize images"""
    data = data.astype('float32')
    data = (data / 255) - 0.5
    return data


def grayscale(data):
    """Grayscale images"""
    imgs = np.ndarray((data.shape[0], 32, 32, 1), dtype=np.uint8)

    for i, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = np.expand_dims(img, axis=2)
        imgs[i] = img

    return imgs


def preprocess(data):
    """Convert to grayscale, histogram equalize, and expand dims"""
    imgs = grayscale(data)
    imgs = center_normalize(imgs)
    return imgs
