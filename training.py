from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from descriptors import fast, edge_histogram

#train path
train_path = '/media/ubuntu/MultimediaAn/ClassificationUV/ROIs/unmerged/training_prueba'

# make a fix file size
fixed_size  = tuple((500,500))

# no of trees for Random Forests
num_tree = 100

# bins for histograms
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same result
seed = 9

# get the training data labels
train_labels = os.listdir(train_path)
foo=0

# sort the training labesl
train_labels.sort()
print(train_labels)

# empty list to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 10

# ittirate the folder to get the image label name

#% time
# lop over the training data sub folder

# features description -1:  Hu Moments

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor -2 Haralick Texture

def fd_haralick(image):
    # conver the image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Ccompute the haralick texture fetature ve tor
    haralic = mahotas.features.haralick(gray).mean(axis=0)
    return haralic

# feature-description -3 Color Histogram

def fd_histogram(image, mask=None):
    # conver the image to HSV colors-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #COPUTE THE COLOR HISTPGRAM
    hist  = cv2.calcHist([image],[0,1,2],None,[bins,bins,bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist,hist)
    # return the histog....
    return hist.flatten()

def get_maximum_feature_vector_size(feature_vectors):
    max = 0
    for fv in feature_vectors:
        if fv.shape[0]>max:
            max = fv.shape[0]
    return max

def complete_feature_vectors(feature_vectors):
    nd_feature_vectors = []
    for fv in feature_vectors:
        difference = max_fv_size - fv.shape[0]
        while difference > 0:
            fv=np.append(fv,[0])
            difference = difference - 1
        nd_feature_vectors.append(fv)
    return nd_feature_vectors

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder

    for file in os.listdir(dir):
        if k<=images_per_class:
            file = dir + "/" + os.fsdecode(file)

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)

            if image is not None:
                image = cv2.resize(image, fixed_size)
                fv_fast = fast.get_features(image).flatten()
                fv_edge_hist = edge_histogram.get_features(image).flatten()
                #fv_hu_moments = fd_hu_moments(image)
                #fv_haralick = fd_haralick(image)
                #fv_histogram = fd_histogram(image)
            # else:
            # print("image not loaded")

            # image = cv2.imread(file)
            # image = cv2.resize(image,fixed_size)

            # Concatenate global features
            global_feature = np.hstack([fv_fast,fv_edge_hist])#.tolist()
            #global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

            i += 1
            k += 1

    print("[STATUS] processed folder: {}".format(current_label))
    j += 1

print("[STATUS] completed Global Feature Extraction...")

#%time
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

max_fv_size = get_maximum_feature_vector_size(global_features)
global_features = complete_feature_vectors(global_features)


# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...{}")
# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")