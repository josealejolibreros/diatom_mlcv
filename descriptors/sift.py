import numpy as np
import cv2
from matplotlib import pyplot as plt

import cv2
import pysift


#img = cv2.imread('/media/ubuntu/DATA/diatoms_patches/ulnaria.tif', 0)
img = cv2.imread('ulnaria.tif', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(img)
print(descriptors)
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img2=cv2.drawKeypoints(img,keypoints,img)

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(img2,cmap = 'gray')
plt.title('SIFT keypoints')

plt.show()