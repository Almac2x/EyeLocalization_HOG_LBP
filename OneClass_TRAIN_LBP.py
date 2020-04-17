from Feature_Descriptors.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import OneClassSVM
from imutils import paths
import argparse
import cv2
import os
import ntpath

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
                help="path to the training images")
ap.add_argument("-e", "--testing", required=False,
                help="path to the tesitng images")
ap.add_argument("-n", "--name", required=False,
                help="name of the model to be saved")
ap.add_argument("-s", "--splits", required=False,
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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = desc.describe(gray,ntpath.basename(imagePath))


    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)


# train a Linear SVM on the data
model = OneClassSVM()

model.fit(data)

image = cv2.imread("dataset/Open_RealSense/s0012_06144_0_0_1_0_1_01.png")
cap2 = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
hist = desc.describe(gray, ntpath.basename(imagePath))

reshape = hist.reshape(1,-1)
Predict = model.predict(reshape)
Func = model.decision_function(reshape)
Score_Samples = model.score_samples(data)

print("Predict" + str(Predict))
print("Func" + str(Func))
#print("Score_Samples" + str(Score_Samples))
