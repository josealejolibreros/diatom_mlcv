import numpy as np
import cv2
from matplotlib import pyplot as plt

# Define the window size
windowsize_r = 64
windowsize_c = 64

winSize = 64
blockSize = (8,8)
blockStride = (4,4)

img = cv2.imread('e-1.tif', 0)

## Detector and extractor
test_image = cv2.resize(img,(512,512))
# Initiate detector
orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE,patchSize=winSize,scaleFactor=1.0,
                     nlevels=1,fastThreshold=10)
orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE,fastThreshold=20,nlevels=10)
for r in range(0,test_image.shape[0], windowsize_r):
    for c in range(0,test_image.shape[1], windowsize_c):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        # find the keypoints and descriptors with SIFT
        keypoints, descriptor = orb.detectAndCompute(window,None)

        frame=cv2.drawKeypoints(window, keypoints,np.array([]), color=(0, 255, 0), flags=0)
        plt.imshow(frame), plt.show()
        foo =0
print(descriptor)