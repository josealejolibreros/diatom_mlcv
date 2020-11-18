from skimage import feature
import numpy as np
import mahotas as mh
import cv2
import imutils
import matplotlib.pyplot as plt

def get_features(image, radius=21):
    # threshold the image
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # initialize the outline image, find the outermost
    # contours (the outline) of the pokemon, then draw
    # it
    outline = np.zeros(image.shape, dtype="uint8")


    cnts = cv2.findContours(img_dilation.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [cnts], -1, 255, -1)
    #c = plt.imshow(outline)
    #plt.title('matplotlib.pyplot.imshow() function Example',
    #          fontweight="bold")
    #plt.show()
    queryFeatures = describe(outline, radius)

    #cv2.imshow("image", image)
    #cv2.imshow("outline", outline)
    #cv2.waitKey(0)
    return queryFeatures

def describe(image, radius):
		# Return the Zernike moments for the image
		# http://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.zernike_moments
		# mahotas.features.zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})
		return mh.features.zernike_moments(image, radius)