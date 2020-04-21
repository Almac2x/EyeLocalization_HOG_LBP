import pickle

from imutils import paths
import cv2, argparse, os
from Descriptors.HOG import HOG
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse

#Command : python train_model_HOG.py --dataset dataset --name Nani --splits 3

# Construct Arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the dataset for training and testing")
ap.add_argument("-n", "--name", required=True,
                help="name of the model to be saved")
ap.add_argument("-s", "--splits", required=True,
                help="number of splits for KFold")
args = vars(ap.parse_args())

# initialize the local binary patterns descriptor along with
# the data and label lists
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["dataset"]):
    # load the image, convert it to grayscale, and describe it

    image = cv2.imread(imagePath)
    cap2 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = HOG.getHOGimage(gray)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)


# %% Train the linear SVM

print(" Training Linear SVM classifier...")
model = LinearSVC(C=1.0)
model.fit(data ,labels)


# Saves the model as a pickle
filename = "HOG_" + str(args["name"]) + "_" + " _KF#" + ".sav"
# pickle.dump(model, open(filename, 'wb'))
with open('Eye_Detection_Model/HOG/' + filename, 'wb') as f:
    pickle.dump(model, f)


for imagePath in paths.list_images(args["dataset"]):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Pli", gray)
    hist = HOG.getHOGimage(gray)
    prediction = model.predict(hist.reshape(1, -1))
    score = model.decision_function(hist.reshape(1,-1))

    print("Predict: {} / Score: {}".format(prediction,score))


    # display the image and the prediction
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 255), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

print(model.score(data,labels))


