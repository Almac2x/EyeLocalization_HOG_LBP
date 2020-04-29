import pickle

from imutils import paths
import cv2, argparse, os
from Descriptors.LBP_HOG import LBP_HOG
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy import sparse


# Command : python train_model_HOG.py --dataset dataset --name Nani --splits 3

def save_Model(model, Score, i):
    # Saves the model as a pickle
    filename = "HOG" + str(args["name"]) + "_" + str(Score) + " _KF#" + str(i) + ".sav"
    # pickle.dump(model, open(filename, 'wb'))
    with open('Eye_Detection_Model/' + filename, 'wb') as f:
        pickle.dump(model, f)


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
    gray = cv2.cvtColor(cap2, cv2.COLOR_BGR2GRAY)
    feat = LBP_HOG.getLBPHOG(gray)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(feat)

# %% Train the linear SVM
print(" Training Linear SVM classifier...")
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

#Save Model
save_Model(model, model.score(data, labels), 2)
