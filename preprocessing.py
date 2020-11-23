import numpy as np
import cv2

def preprocess_image(img):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.4) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    equalised = cv2.equalizeHist(res)
    converted_img = cv2.cvtColor(equalised, cv2.COLOR_GRAY2BGR)
    img_preprocessed = cv2.fastNlMeansDenoisingColored(converted_img,None,20,20,15,21)
    return img_preprocessed