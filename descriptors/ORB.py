import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('ulnaria.tif', 0)

## Detector and extractor

# Initiate detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
keypoints, descriptor = orb.detectAndCompute(img,None)
print(descriptor)