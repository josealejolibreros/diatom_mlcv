from skimage import feature
import numpy as np
import mahotas as mh
import cv2
import imutils
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix,greycoprops
def get_features(image):
    sub_window_size_r = int(64 / 2)
    sub_window_size_c = int(64 / 2)
    dissimilarity_desc = []
    contrast_desc = []
    homogeneity_desc = []
    asm_desc = []
    energy_desc = []
    correlation_desc = []
    for r in range(0, image.shape[0] - 1, sub_window_size_r):
        for c in range(0, image.shape[1] - 1, sub_window_size_c):
            window = image[r:r + sub_window_size_r, c:c + sub_window_size_c]
            g = greycomatrix(window, distances=[5], angles=[0], levels=256,
                             symmetric=True, normed=True)
            dissimilarity = greycoprops(g, 'dissimilarity')
            dissimilarity_desc.append(dissimilarity[0, 0])

            contrast = greycoprops(g, 'contrast')
            contrast_desc.append(contrast[0, 0])

            homogeneity = greycoprops(g, 'homogeneity')
            homogeneity_desc.append(homogeneity[0, 0])

            asm = greycoprops(g, 'ASM')
            asm_desc.append(asm[0, 0])

            energy = greycoprops(g, 'energy')
            energy_desc.append(energy[0, 0])

            correlation = greycoprops(g, 'correlation')
            correlation_desc.append(correlation[0, 0])

    grey_coocurrence_props_descriptors = np.concatenate([dissimilarity_desc,contrast_desc,homogeneity_desc,
                                                         asm_desc,energy_desc,correlation_desc])


    return grey_coocurrence_props_descriptors
