# import the necessary packages
from skimage import feature
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

def get_features(image, eps=1e-7,numPoints=24, radius=8):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
        radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2))
    # normalize the histogram
    #hist = hist.astype("float")
    #hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist



for r in range(0,test_image.shape[0], windowsize_r):
    for c in range(0,test_image.shape[1], windowsize_c):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        hist = get_features(window, eps=1e-7,numPoints=24, radius=8)
        foo = 0


