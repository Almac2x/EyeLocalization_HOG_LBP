import cv2
import os

from imutils import paths

from pyimagesearch.helpers import sliding_window

# load the image and define the window width and height
(winW, winH) = (64, 64)

# Change Dataset
dataset_path = "../Yale_Faces"
image_path_save_negative = "../Dataset/Negative"
image_path_save_eyes_right = "../Dataset/Eyes_Right"
image_path_save_eyes_left = "../Dataset/Eyes_Left"


# os.mkdir(new_path_save)

def scan_image(image, name):
    # loop over the image pyramid
    count = 1
    for (x, y, window) in sliding_window(image, stepSize=64, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        if count == 1:
            crop_img = image[y + 10:y + 10 + winH, x + 10:x + 10 + winW]
        elif count == 3:
            crop_img = image[y + 10:y + 10 + winH, x - 10:x - 10 + winW]
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
            cv2.imwrite('%s/%s-%s.png' % (image_path_save_negative, name, count), crop_img)
            print("image save to" + image_path_save_negative)

        count += 1


data = []
labels = []

# loop over the training images

for imagePath in paths.list_images(dataset_path):
    # load the image, convert it to grayscale, and describe it

    image = cv2.imread(imagePath)
    print("Image Path: {} \nImage Height: {} \nImage Width: {} ".format(imagePath.split(os.path.sep)[-1], image.shape[0], image.shape[1]))
    resize = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    scan_image(gray, os.path.splitext(imagePath.split(os.path.sep)[-1])[0])

    # extract the label from the image path, then update the
    # label and data lists
