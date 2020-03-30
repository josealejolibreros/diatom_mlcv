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
from sklearn.metrics import classification_report

from descriptors import fast, edge_histogram

# make a fix file size
fixed_size  = tuple((500,500))

# no of trees for Random Forests
num_tree = 100

# train_test_split size
test_size = 0.10

# seed for reproducing same result
seed = 9


# import the feature vector and trained labels

h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)


# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)


# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

#print(clf.fit(trainDataGlobal, trainLabelsGlobal))

clf_pred = clf.predict(trainDataGlobal)
#clf_pred = clf.predict(global_feature.reshape(1,-1))[0]
print(classification_report(trainLabelsGlobal,clf_pred))
#print(confusion_matrix(trainLabelsGlobal,clf_pred))

#print(clf.predict(trainDataGlobal))

#print(clf.predict(global_feature.reshape(1,-1))[0])


# path to test data
test_path = "/media/ubuntu/MultimediaAn/ClassificationUV/ROIs/unmerged/test_prueba"
test_labels = os.listdir(test_path)
test_labels.sort()

# loop through the test images
# for file in glob.glob(test_path + "/*.jpg"):
for test_name in os.listdir(test_path):
    dir = os.path.join(test_path, test_name)
    for file in os.listdir(dir):
        file = dir + "/" + os.fsdecode(file)
        print(file)

        # read the image
        image = cv2.imread(file)

        # resize the image
        image = cv2.resize(image, fixed_size)

        # Global Feature extraction
        fv_fast = fast.get_features(image).flatten()
        fv_edge_hist = edge_histogram.get_features(image).flatten()
        #fv_hu_moments = fd_hu_moments(image)
        #fv_haralick = fd_haralick(image)
        #fv_histogram = fd_histogram(image)

        # Concatenate global features
        global_feature = np.hstack([fv_fast, fv_edge_hist])  # .tolist()
        #global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        n_features = global_features.shape[1]
        new_global_feature = np.zeros((1,n_features))[0]
        for i in range(global_feature.shape[0]):
            new_global_feature[i]=global_feature[i]
        new_global_feature=np.nan_to_num(new_global_feature)

        # predict label of test image
        prediction = clf.predict(new_global_feature.reshape(1, -1))[0]

        # show predicted label on image
        cv2.putText(image, test_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()