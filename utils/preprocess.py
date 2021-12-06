import cv2
import numpy as np

def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray*(180/np.max(gray))
    gray = gray.astype(np.uint8)
    high = highpass(gray,21)
    high[high>120] = 255
    high[high<=120] = 0
    return high
