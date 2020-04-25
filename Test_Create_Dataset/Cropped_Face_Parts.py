from sklearn.model_selection import KFold

from Descriptors.HOG import HOG
from Descriptors.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import argparse
import cv2
import os
import pickle, statistics
import numpy as np
import ntpath

from imutils import paths

from pyimagesearch.helpers import pyramid, sliding_window

# load the image and define the window width and height
(winW, winH) = (64,64)

#Change Dataset
dataset_path = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\CroppedYale_ConvertedPNG"
image_path_save = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\Cropped_64x64\.."

def scan_image(image, name):
    # loop over the image pyramid
    count=1
    os.mkdir(image_path_save + name)
    for (x, y, window) in sliding_window(image, stepSize=64, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        crop_img = image[y:y + winH, x:x + winW]

        # Writes the cropped to disk

        new_image_path_save = image_path_save++name

        cv2.imwrite('%s/%s-%s.png' % (new_image_path_save, str(name+count), count), crop_img)
        print("image save")

        #Shows How Sliding windows works
        count += 1
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 255), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(0)


data = []
labels = []



# loop over the training images

for imagePath in paths.list_images(dataset_path):
    # load the image, convert it to grayscale, and describe it

    image = cv2.imread(imagePath)
    print("Image Path: {} \n Image Height: {} \n Image Width: {} ".format(imagePath.split(os.path.sep)[-1],
                                                                          image.shape[0], image.shape[1]))
    resize = cv2.resize(image, (192,192), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    cv2.imshow(str(imagePath), resize)

    scan_image(gray, imagePath.split(os.path.sep)[-1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # extract the label from the image path, then update the
    # label and data lists
