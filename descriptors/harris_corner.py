import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/media/ubuntu/DATA/diatoms_patches/ulnaria.tif', 0)
corn = img.copy()

dst = cv2.cornerHarris(img,3,5,0.22)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
corn[dst>0.01*dst.max()]=0

plt.figure(figsize=(15,12))

plt.subplot(121)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(corn,cmap = 'gray')
plt.title('Corner Image')

plt.show()