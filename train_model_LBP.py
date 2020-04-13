# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from Feature_Descriptors.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")
ap.add_argument("-e", "--testing", required=False,
                help="path to the tesitng images")
ap.add_argument("-n", "--name", required=True,
                help="name of the model to be saved")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    cap2 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
DataTrain, DataTest, LabelTrain, LabelTest = train_test_split(data, labels, test_size=0.2, random_state=42)

# Cross Validation Score
# Cross_Score = cross_val_score(model,data,labels,cv=3)

i = 0
for Cross_Score in cross_val_score(model, data, labels, cv=3):
    i = i + 1
    print(Cross_Score)

# Trains the model
#model.fit(DataTrain, LabelTrain)
#predictions = model.predict(DataTest)

# KFold
kf = KFold(n_splits=2, random_state=None, shuffle=False)

i=0
for train_index, test_index in kf.split(data):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    i = i+1
    print("KFold: "+i)
    model.fit(x_train,y_train)
    # Check the score of the Model
    print('Test Accuracy of SVC = ', round(model.score(DataTest, LabelTest), 4))

    # Saves the model as a pickle
    filename = str(args["name"]) + ".sav"
    pickle.dump(model, open(filename, 'wb'))


# Plots the model
#plt.scatter(LabelTest, predictions)
#plt.xlabel("True Value")
#plt.ylabel("Predictions")
#plt.show()




# loop over the testing images
for imagePath in paths.list_images(args["testing"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Pli", gray)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))

    # display the image and the prediction
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
