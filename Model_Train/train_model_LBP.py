# USAGE
# python recognize.py --training images/training --testing images/testing

# import the necessary packages
from sklearn.model_selection import KFold


from Feature_Descriptors.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import pickle, statistics
import numpy as np
import ntpath

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")
ap.add_argument("-e", "--testing", required=False,
                help="path to the tesitng images")
ap.add_argument("-n", "--name", required=True,
                help="name of the model to be saved")
ap.add_argument("-s", "--splits", required=True,
                help="number of splits for KFold")
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
    hist = desc.describe(gray,ntpath.basename(imagePath))


    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)


# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)


# Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
#DataTrain, DataTest, LabelTrain, LabelTest = train_test_split(data, labels, test_size=0.2, random_state=42)

# Cross Validation Score
# Cross_Score = cross_val_score(model,data,labels,cv=3)

#i = 0
#for Cross_Score in cross_val_score(model, data, labels, cv=3):
#    i = i + 1
#    print("KFold: " + str(i))
#    print("Cross_Validation Score: " + str(Cross_Score))


# Trains the model
# model.fit(DataTrain, LabelTrain)
# predictions = model.predict(DataTest)

# KFold
kf = KFold(n_splits=int(args["splits"]), random_state=None, shuffle=False)
Cross_Validation_Score = []

i = 0
for train_index, test_index in kf.split(data):
    print("*****************************************")
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = np.array(data)[train_index.astype(int)], np.array(data)[test_index.astype(int)]
    y_train, y_test = np.array(labels)[train_index.astype(int)], np.array(labels)[test_index.astype(int)]
    i = i + 1
    print("KFold: " + str(i))
    model.fit(x_train, y_train)

    # Check the score of the Model
    Score = round(model.score(x_test, y_train), 4)
    print('Test Accuracy of SVC = ', Score)
    Cross_Validation_Score.append(Score)

    # Saves the model as a pickle
    filename = "LBP_"+str(args["name"]) + "_" + str(Score) + " _KF#" + str(i) + ".sav"
    # pickle.dump(model, open(filename, 'wb'))
    with open('Eye_Detection_Model/' + filename, 'wb') as f:
        pickle.dump(model, f)

    image = cv2.imread("../images/Blando_1.jpg")
    cap2 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray, "images/patrick_bateman.jpg")
    #Test
    hist2 = hist.reshape(1,-1)
    print(model.predict(hist2))
    print(model.decision_function(hist2))

print("Cross Validation Median: " + str(statistics.median(Cross_Validation_Score)))


# Plots the model
# plt.scatter(y_test, model.predict(x_test))
# plt.xlabel("True Value")
# plt.ylabel("Predictions")
# plt.show()


# loop over the testing images
# for imagePath in paths.list_images(args["testing"]):
#     # load the image, convert it to grayscale, describe it,
#     # and classify it
#     image = cv2.imread(imagePath)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Pli", gray)
#     hist = desc.describe(gray)
#     prediction = model.predict(hist.reshape(1, -1))
#
#     # display the image and the prediction
#     cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.0, (0, 0, 255), 3)
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)

#Command
#python train_model_LBP.py --training C:\Users\bland\Desktop\Thesis\Datasets\Oversize_Dataset_Test --testing dataset --name Oversize_Dataset --s 5
