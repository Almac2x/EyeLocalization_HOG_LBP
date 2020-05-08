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
(winW, winH) = (64, 64)

# Change Dataset
dataset_path = r"D:\Chrome Downloads\Thesis Downloads\BioID\PNG"
image_path_save = r"D:\Chrome Downloads\Thesis Downloads\BioID\Negative"  # Negative
image_path_save_eyes = r"D:\Chrome Downloads\Thesis Downloads\BioID\Eyes"
image_path_eyes_loc = r"D:\Chrome Downloads\Thesis Downloads\BioID\EyePoints\\"
# image_path_save_eyes_left = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Left"


for imagePath in paths.list_images(dataset_path):

    # Initializes Eye Location Array
    Eye_loc = np.array([])

    # Gets the image name of
    image_name = os.path.splitext(imagePath.split(os.path.sep)[-1])[0]

    image = cv2.imread(imagePath)
    print("Image Path: {} \n Image Height: {} \n Image Width: {} ".format(image_name,
                                                                          image.shape[0], image.shape[1]))
    # rea
    eye_loc_file = open(image_path_eyes_loc + image_name + ".eye", "r")
    eye_loc_string = eye_loc_file.read()

    result = eye_loc_string.split("\t")

    print(result)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # loop over the bounding boxes for each image and draw them
    for (startX, startY, endX, endY) in Eye_loc:
        cv2.rectangle(gray, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Face", gray)
    cv2.waitKey(0)

    # scan_image(gray, os.path.splitext(imagePath.split(os.path.sep)[-1])[0])
