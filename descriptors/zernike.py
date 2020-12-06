from skimage import feature
import numpy as np
import mahotas as mh
import cv2
import imutils
import matplotlib.pyplot as plt

def get_features(image, radius=21):

    # threshold the image

    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv2.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.4, abs_grad_y, 0.4, 5)

    thresh = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    outline = np.zeros(image.shape, dtype="uint8")

    cnts = cv2.findContours(img_dilation.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts)>0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        cv2.drawContours(outline, [cnts], -1, 255, -1)
        #c = plt.imshow(outline)
        #plt.title('matplotlib.pyplot.imshow() function Example',
        #          fontweight="bold")
        #plt.show()
        queryFeatures = describe(outline, radius)

        #cv2.imshow("image", image)
        #cv2.imshow("outline", outline)
        #cv2.waitKeyEx(0)
    else:
        print("no hubo contornos zernike")
        queryFeatures = np.zeros(25)
    return queryFeatures

def describe(image, radius):
		# Return the Zernike moments for the image
		# http://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.zernike_moments
		# mahotas.features.zernike_moments(im, radius, degree=8, cm={center_of_mass(im)})
		return mh.features.zernike_moments(image, radius)