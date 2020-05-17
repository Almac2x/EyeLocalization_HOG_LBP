from sklearn.svm import LinearSVC
import argparse
import cv2
import os
import pickle, statistics
import numpy as np
import ntpath

from Face_Detection import Face_Detection
from imutils import paths

from pyimagesearch.helpers import pyramid, sliding_window


def scan_image(image, name, new_path_save,image_path_save_eyes_right,image_path_save_eyes_left):
    (winH,winW) = (64,64)



    # loop over the image pyramid
    count = 1
    for (x, y, window) in sliding_window(image, stepSize=64, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        if count == 1:
            crop_img = image[y+10:y+10 + winH, x+10:x+10 + winW]
        elif count == 3:
            crop_img = image[y+10:y+10 + winH, x-10:x-10 + winW]
        else:
            crop_img = image[y:y + winH, x:x + winW]



        # Writes the cropped to disk
        if count == 1:
            cv2.imwrite('%s/%s-%s.png' % (image_path_save_eyes_right, name, count), crop_img)
            print("image save to" + image_path_save_eyes_right)
        elif count == 3:
            cv2.imwrite('%s/%s-%s.png' % (image_path_save_eyes_left, name, count), crop_img)
            print("image save to" + image_path_save_eyes_left)

        else:
            cv2.imwrite('%s/%s-%s.png' % (new_path_save, name, count), crop_img)
            #qprint("image save")

        # Shows How Sliding windows works
        count += 1
        #clone = image.copy()
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 255), 2)
        #cv2.imshow("Window", clone)
        #cv2.waitKey(0)





# load the image and define the window width and height
(winW, winH) = (24, 24-10)

# Change Dataset
# dataset_path = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Bio_ID_Face"
# image_path_save = r"D:\Chrome Downloads\Thesis Downloads\BioID\Negative"  # Negative
# image_path_save_eyes_right = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Right_Eye"
# image_path_save_eyes_left = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Left_Eye"
# image_path_eyes_loc = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Eye_Loc\\"
#
# sliding_image_save_neg = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Bio_ID_SlidingWindow\Negative"
# sliding_path_save_eyes_right = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Bio_ID_SlidingWindow\Eye_Right"
# sliding_path_save_eyes_left = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Bio_ID_SlidingWindow\Eye_Left"

dataset_path = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\BioID_Converted_PNG"
image_path_save = r"D:\Chrome Downloads\Thesis Downloads\BioID\Negative"  # Negative
image_path_save_eyes_right = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Right"
image_path_save_eyes_left = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Left"
image_path_eyes_loc = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\BioID-FD-Eyepos-V1.2\\"

sliding_image_save_neg = r"D:\Documents\Chrome Downloads\Thesis Download\Datasets\BioID\Bio_ID_SlidingWindow\Negative"
sliding_path_save_eyes_right = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Right"
sliding_path_save_eyes_left = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Left"

# image_path_save_eyes_left = r"C:\Users\Pili\PycharmProjects\EyeLocalization_HOG_LBP\Test_Create_Dataset\Eyes_Left"

Face_Detection = Face_Detection()


for imagePath in paths.list_images(dataset_path):

    # Initializes Eye Location Array
    Eye_loc = []

    # Gets the image name of
    image_name = os.path.splitext(imagePath.split(os.path.sep)[-1])[0]

    image = cv2.imread(imagePath)
    print("Image Path: {} \n Image Height: {} \n Image Width: {} ".format(image_name,
                                                                          image.shape[0], image.shape[1]))
    # rea
    # eye_loc_file = open(image_path_eyes_loc + image_name + ".eye", "r")
    # eye_loc_string = eye_loc_file.read()

    # Splits file to array
    # result = eye_loc_string.split()

    # Eye Starting Location
    # Lx = int(result[4])
    # Ly = int(result[5])
    # Rx = int(result[6])
    # Ry = int(result[7])
    # Inserts Left Eye Loc
    # Eye_loc.extend([[Lx - winW, Ly - winH, Lx + (winW), Ly + winH]])
    # # Inserts Left Eye Loc
    # Eye_loc.extend([[Rx - winW, Ry - winH, Rx + winW, Ry + winH]])

    # print(Eye_loc)


    #Gets the images Face Location
    Face_Loc = Face_Detection.getFace(image)

    if len(Face_Loc) == 4:
        (startX, startY, endX, endY) = Face_Loc.astype("int")
        cv2.rectangle(image, (startX, startY+40), (endX, endY),
                      (0, 0, 255), 2)

        # Perform Sliding Window
        roi = image[startY+40:int(endY), startX:endX]
        roi_resize = roi_resize = cv2.resize(roi, (192, 192), interpolation=cv2.INTER_AREA)

        scan_image(roi_resize, image_name, sliding_image_save_neg, sliding_path_save_eyes_right,
                   sliding_path_save_eyes_left)

    else:
        print("No face Detected")


    # Crops Left and Right Eye
    # crop_img_left_eye = image[Eye_loc[0][1]:Eye_loc[0][3], Eye_loc[0][0]:Eye_loc[0][2]]
    #
    # crop_img_right_eye = image[Eye_loc[1][1]:Eye_loc[1][3], Eye_loc[1][0]:Eye_loc[1][2]]



    # Writes left and Right eye
    # cv2.imwrite('%s/%s.png' % (image_path_save_eyes_right, image_name + "_right"), crop_img_right_eye)
    # print("image save to" + image_path_save_eyes_right)
    #
    # cv2.imwrite('%s/%s.png' % (image_path_save_eyes_left, image_name + "_left"), crop_img_left_eye)
    # print("image save to" + image_path_save_eyes_right)

    # cv2.imshow("Left_Eye", crop_img_left_eye)
    # cv2.imshow("Right_Eye", crop_img_right_eye)
    cv2.imshow("Face", image)

    cv2.waitKey(0)

