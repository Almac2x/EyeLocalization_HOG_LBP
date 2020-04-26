import pickle

from imutils import paths
import cv2, argparse, os
from Descriptors.HOG import HOG
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy import sparse

# Command : python train_model_HOG.py --dataset dataset --name Nani --splits 3

def save_Model(model,Score,i):
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = HOG.getHOGimage(gray)

    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)

# %% Train the linear SVM

print(" Training Linear SVM classifier...")
model = LinearSVC(C=1.0)
model.fit(data,labels)
save_Model(model,model.score(data,labels),2)




#Splitting
# KFold
#kf = KFold(n_splits=int(args["splits"]), random_state=None, shuffle=False)
#Cross_Validation_Score = []

#i = 0
#for train_index, test_index in kf.split(data):
#    print("*****************************************")
#    #print("TRAIN:", train_index, "TEST:", test_index)
#    x_train, x_test = np.array(data)[train_index.astype(int)], np.array(data)[test_index.astype(int)]
#    y_train, y_test = np.array(labels)[train_index.astype(int)], np.array(labels)[test_index.astype(int)]
#    i = i + 1
#   print("KFold: " + str(i))
#    model.fit(x_train, y_train)

#    # Check the score of the Model
#    Score = round(model.score(x_test, y_train), 4)
#    print('Test Accuracy of SVC = ', Score)
#    Cross_Validation_Score.append(Score)






