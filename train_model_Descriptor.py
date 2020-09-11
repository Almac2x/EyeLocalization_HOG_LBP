# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from sklearn.model_selection import KFold

from Descriptors.localbinarypatterns import LocalBinaryPatterns
from Descriptors.LBP_HOG import LBP_HOG
from Descriptors.HOG import HOG
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from imutils import paths
import argparse
import cv2
import os
import pickle
import numpy as np
import ntpath

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=False,
                help="path to the training images")
ap.add_argument("-n", "--name", required=False,
                help="name of the model to be saved")
ap.add_argument("-s", "--splits", required=False,
                help="number of splits for KFold")
ap.add_argument("-m", "--model", required=False,
                help="what kind of model to train (LBP, HOG, LBP_HOG)")
args = vars(ap.parse_args())

args["name"] = "HOG_NEW_3_KFOLD_COPYPASTE"
args["splits"] = "3"
args["model"] = "HOG"
args["training"]= r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Yale"

# initialize the local binary patterns descriptor along with
# the data and label lists
desc_LBP = LocalBinaryPatterns()
desc_LBP_HOG = LBP_HOG("bruh")

data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
    # load the image, convert it to grayscale, and describe it

    image = cv2.imread(imagePath)
    cap2 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(cap2, cv2.COLOR_BGR2GRAY)

    if args["model"] == "LBP":
        desc = desc_LBP.describe(gray)
    elif args["model"] == "HOG":
        desc = HOG.getHOGimage(gray)
    elif args["model"] == "LBP_HOG":
        desc = desc_LBP_HOG.getLBPHOG(gray, 'dsad', 'vcxv')

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    print(imagePath.split(os.path.sep)[-2])
    data.append(desc)

# train a Linear SVM on the data

# X_train, X_test, y_train, y_test = train_test_split(
#     data, labels, test_size=0.2, random_state=42)
#
model = LinearSVC(C=100.0, random_state=42,dual=False)
# model.fit(X_train, y_train)
# print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4))
 #Saves the model as a pickle
# filename = str(args["name"]) + "_" + " _KF#" + str(1) + ".sav"
# pickle.dump(model, open(filename, 'wb'))
# with open('Eye_Detection_Model/' + filename, 'wb') as f:
#         pickle.dump(model, f)


# KFold
kf = KFold(n_splits=int(args["splits"]), random_state=None, shuffle=True)
Cross_Validation_Score = []

# For each KFOLD will be trained and save
i = 0
for train_index, test_index in kf.split(data):
    print("*****************************************")
    # print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = np.array(data)[train_index.astype(int)], np.array(data)[test_index.astype(int)]
    y_train, y_test = np.array(labels)[train_index.astype(int)], np.array(labels)[test_index.astype(int)]
    i = i + 1
    print("KFold: " + str(i))
    model.fit(x_train, y_train)

    # Check the score of the Model
    Score = round(model.score(x_test, y_test), 4)
    print(Score)
    print('Test Accuracy of SVC = ', Score)
    Cross_Validation_Score.append(Score)
    #
    #     # Saves the model as a pickle
    filename = str(args["name"]) + "_" + str(Score) + " _KF#" + str(i) + ".sav"
    pickle.dump(model, open(filename, 'wb'))
    with open('Eye_Detection_Model/' + filename, 'wb') as f:
        pickle.dump(model, f)
print(Cross_Validation_Score)
print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(Cross_Validation_Score), np.std(Cross_Validation_Score) * 2))
