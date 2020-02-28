import cv2
from matplotlib import pyplot as plt

img = cv2.imread('ulnaria.tif', 0)

## Detector
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(img)

## Extractor
br = cv2.BRISK_create();
keypoints, descriptor = br.compute(img,  keypoints)
print(descriptor)

img2 = img.copy()
for marker in keypoints:
    img2 = cv2.drawMarker(img2, tuple(int(i) for i in marker.pt), color=(0, 255, 255))

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(img2,cmap = 'gray')
plt.title('Blob Detection Image')

plt.show()