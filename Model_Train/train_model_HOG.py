from imutils import paths
from skimage.feature import hog
import cv2,time,numpy,argparse,os

from Feature_Descriptors.HOG import HOG

#Construct Arguments

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



    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    #data.append(hist)

print(labels)