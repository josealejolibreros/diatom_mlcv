import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('ulnaria.tif', 0)


def get_features(img, threshold=9):

    ## Detector

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create(threshold)
    # find and draw the keypoints
    keypoints = fast.detect(img, None)

    ## Extractor
    br = cv2.BRISK_create();
    keypoints, descriptor = br.compute(img,  keypoints)
    descriptor = np.asarray(descriptor)
    #print(descriptor)

    return descriptor



'''
img2 = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(keypoints)))
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
keypoints = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(keypoints)))
img3 = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0))

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(img2,cmap = 'gray')
plt.title('Fast keypoints')

plt.subplot(221)
plt.imshow(img3,cmap = 'gray')
plt.title('Fast Keypoints without nonmaxSuppression')

plt.show()
'''