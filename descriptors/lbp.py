from skimage import feature
import numpy as np

def get_features(image, eps=1e-7,numPoints=24, radius=8):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
        radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, numPoints + 3),
        range=(0, numPoints + 2), density=True)
    # normalize the histogram
    #hist = hist.astype("float")
    #hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist