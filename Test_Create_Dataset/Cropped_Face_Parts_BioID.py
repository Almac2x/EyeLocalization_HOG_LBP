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
(winW, winH) = (24, 24-10)

# Change Dataset
dataset_path = r"D:\Chrome Downloads\Thesis Downloads\BioID\PNG"
image_path_save = r"D:\Chrome Downloads\Thesis Downloads\BioID\Negative"  # Negative
image_path_save_eyes_right = r"D:\Chrome Downloads\Thesis Downloads\BioID\Eyes\Eyes_Right"
image_path_save_eyes_left = r"D:\Chrome Downloads\Thesis Downloads\BioID\Eyes\Eyes_Left"
image_path_eyes_loc = r"D:\Chrome Downloads\Thesis Downloads\BioID\EyePoints\\"
# image_path_save_eyes_left = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Left"


for imagePath in paths.list_images(dataset_path):

    # Initializes Eye Location Array
    Eye_loc = []

    # Gets the image name of
    image_name = os.path.splitext(imagePath.split(os.path.sep)[-1])[0]

    image = cv2.imread(imagePath)
    print("Image Path: {} \n Image Height: {} \n Image Width: {} ".format(image_name,
                                                                          image.shape[0], image.shape[1]))
    # rea
    eye_loc_file = open(image_path_eyes_loc + image_name + ".eye", "r")
    eye_loc_string = eye_loc_file.read()

    # Splits file to array
    result = eye_loc_string.split()

    # Eye Starting Location
    Lx = int(result[4])
    Ly = int(result[5])
    Rx = int(result[6])
    Ry = int(result[7])
    # Inserts Left Eye Loc
    Eye_loc.extend([[Lx - winW, Ly - winH, Lx + (winW), Ly + winH]])
    # Inserts Left Eye Loc
    Eye_loc.extend([[Rx - winW, Ry - winH, Rx + winW, Ry + winH]])

    print(Eye_loc)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crops Left and Right Eye
    crop_img_left_eye = gray[Eye_loc[0][1]:Eye_loc[0][3], Eye_loc[0][0]:Eye_loc[0][2]]

    crop_img_right_eye = gray[Eye_loc[1][1]:Eye_loc[1][3], Eye_loc[1][0]:Eye_loc[1][2]]

    ## loop over the bounding boxes for each image and draw them
   # for (startX, startY, endX, endY) in np.array(Eye_loc):
    #    cv2.rectangle(gray, (startX, startY), (endX, endY), (0, 255, 0), 2)



    # Writes left and Right eye
    cv2.imwrite('%s/%s.png' % (image_path_save_eyes_right, image_name + "_right"), crop_img_right_eye)
    print("image save to" + image_path_save_eyes_right)

    cv2.imwrite('%s/%s.png' % (image_path_save_eyes_left, image_name + "_left"), crop_img_left_eye)
    print("image save to" + image_path_save_eyes_right)

    cv2.imshow("Left_Eye", crop_img_left_eye)
    cv2.imshow("Right_Eye", crop_img_right_eye)
    cv2.imshow("Face", gray)

