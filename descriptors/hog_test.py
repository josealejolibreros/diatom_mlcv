import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import data, exposure
import skimage.feature
image = cv2.imread("e-1.tif",0)
size = (512,512)
test_image = cv2.resize(image,size)

grey_levels = 256
# Generate a test image
#test_image = numpy.random.randint(0,grey_levels, size=(11,11))

# Define the window size
windowsize_r = 64
windowsize_c = 64






winSize = (16,16)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (2,2)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

# Crop out the window and calculate the histogram
for r in range(0,test_image.shape[0], windowsize_r):
    for c in range(0,test_image.shape[1], windowsize_c):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        #hist = numpy.histogram(window,bins=grey_levels)
        #hog = cv2.HOGDescriptor()
        #hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
        #                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
        #compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (8,8)
        padding = (8,8)
        locations = ((10,20),)

        fd, hog_image = skimage.feature.hog(window, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #hist = hog.compute(window,winStride,padding,locations)
        #frame = cv2.drawKeypoints(window, keypoints, np.array([]), color=(0, 255, 0), flags=0)
        #plt.imshow(frame), plt.show()
        ax1.axis('off')
        ax1.imshow(window, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
        foo=0


#fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                    cells_per_block=(1, 1), visualize=True, multichannel=True)
foo=0