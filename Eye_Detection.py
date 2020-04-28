# Website https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import cv2, pickle, os
from Descriptors.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.helpers import pyramid, sliding_window
from Descriptors.HOG import HOG
import numpy as np


class Eyes:
    def __init__(self, Descriptor):
        # Chooses what descriptor to use
        self.Descriptor = Descriptor
        if Descriptor == "LBP":
            self.Model_Path = "Eye_Detection_Model/LBP_Yale_64_Center_ _KF#0.sav"
        elif Descriptor == "HOG":
            self.Model_Path = "Eye_Detection_Model/NewHOGnani7.sav"

        # Loads the model to be used
        self.loaded_model = pickle.load(open(self.Model_Path, 'rb'))

    def getEyes(self, image):
        # Starts of Eye Detection

        # Initializes box array to store eye locations
        Eye_Box_Loc = []

        # load the image and define the window width and height
        (winW, winH) = (64, 64)

        if (self.Descriptor == "LBP"):
            # initialize the local binary patterns descriptor along with
            desc = LocalBinaryPatterns(24, 8)

        # Write a place to put
        # image_path = "Test_Create_Dataset/"
        # if not os.path.isdir(image_path):
        #   os.mkdir(image_path)

        # Converts to BGR2GRAY
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # count = 1

        # loop over the image pyramid
        # for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(image, stepSize=16, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            clone = image.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)

            crop_img = gray[y:y + winH, x:x + winW]

            Box = [x, y, x + winW, y + winH]
            # IMG_Copy = resized.copy()

            # Writes the cropped to disk
            # cv2.imwrite('%s/%s.png' % (image_path, count), crop_img)
            # print("image save")
            # count += 1
            # See how it slides
            # cv2.rectangle(IMG_Copy, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            # cv2.imshow("Window", IMG_Copy)

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # Describes the image
            if (self.Descriptor == "LBP"):
                hist = desc.describe(crop_img, "Frame")
            elif (self.Descriptor == "HOG"):
                hist = HOG.getHOGimage(crop_img)

            # Loads Prediction Model
            reshape_lbp = hist.reshape(1, -1)
            prediction = self.loaded_model.predict(reshape_lbp)

            confidence_level = self.loaded_model.decision_function(reshape_lbp)
            if (self.Descriptor == "LBP"):
                Eye_Open_Confidence_Level = confidence_level[0][1] * 100
                if prediction[0] == "Eyes_Left" or prediction[0] == "Eyes_Right":
                    Eye_Box_Loc.append(Box)

            elif (self.Descriptor == "HOG"):
                Eye_Open_Confidence_Level = confidence_level[0] * 100
                if prediction[0] == "Eyes":
                    Eye_Box_Loc.append(Box)

            print(prediction[0])
            print(Eye_Open_Confidence_Level)

            # print("Status: ".format(prediction[0]))
            # print("Eye_Open Confidenve Level: {:.2f}".format(Eye_Open_Confidence_Level))
            # print("Confidence Level: {}".format(confidence_level))

            # display the image and the prediction
            # XY = (x, y + winH)
            # cv2.putText(image, prediction[0], XY, cv2.FONT_HERSHEY_SIMPLEX,
            #           1.0, (0, 0, 255), 3)

            # Wait Key for Show
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)

        # cv2.destroyAllWindows()

        return np.array(Eye_Box_Loc)
