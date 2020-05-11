# Website https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import cv2, pickle, os

from Descriptors.LBP_HOG import LBP_HOG
from Descriptors.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.helpers import pyramid, sliding_window
from Descriptors.HOG import HOG
import numpy as np


class Eyes:
    def __init__(self, Descriptor):
        # Chooses what descriptor to use
        self.Descriptor = Descriptor
        if Descriptor == "LBP":
            self.Model_Path = "Eye_Detection_Model/Aptina/LBP_Aptina_0.9864 _KF#2.sav"
        elif Descriptor == "HOG":
            self.Model_Path = "Eye_Detection_Model/Aptina/HOGAptinaHOG_0.958029197080292 _KF#2.sav"
        elif Descriptor == "LBP_HOG":
            self.Model_Path = "LBP_HOG_LBP_reTrained_ _KF#1.sav"
        # Loads the model to be used
        self.loaded_model = pickle.load(open(self.Model_Path, 'rb'))

    def getEyes(self, image):
        # Starts of Eye Detection

        # Initializes box array to store eye locations
        Eye_Box_Loc = []

        # EyeBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
        #                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
        # embedder.setInput(EyeBlob)
        # vec = embedder.forward()


        # load the image and define the window width and height
        (winW, winH) = (64, 64)

        if (self.Descriptor == "LBP"):
            # initialize the local binary patterns descriptor along with
            desc = LocalBinaryPatterns()
        elif (self.Descriptor == "LBP_HOG"):
            desc = LBP_HOG("bruh")

        # Converts to BGR2GRAY
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # Describes the image
            if (self.Descriptor == "LBP"):
                hist = desc.describe(crop_img)
            elif (self.Descriptor == "HOG"):
                hist = HOG.getHOGimage(crop_img)
            elif self.Descriptor == "LBP_HOG":
                hist = desc.getLBPHOG(crop_img)

            # Loads Prediction Model
            reshape_lbp = hist.reshape(1, -1)
            prediction = self.loaded_model.predict(reshape_lbp)
            # self.loaded_model.classes
            confidence_level = self.loaded_model.decision_function(reshape_lbp)

            if (self.Descriptor == "LBP"):
                Eye_Open_Confidence_Level = confidence_level[0][1] * 100
                if prediction[0] == "Eye_Open" and Eye_Open_Confidence_Level > 90:
                    Eye_Box_Loc.append(Box)

            elif (self.Descriptor == "HOG"):
                Eye_Open_Confidence_Level = confidence_level[0] * 100
                if prediction[0] == "Eyes":
                    Eye_Box_Loc.append(Box)

            elif self.Descriptor == "LBP_HOG":
                Eye_Open_Confidence_Level = confidence_level[0] * 100
                if prediction[0] == "Eyes_Left" or prediction[0] == "Eyes_Right":
                    Eye_Box_Loc.append(Box)

            print(prediction[0])
            print(Eye_Open_Confidence_Level)

        return np.array(Eye_Box_Loc)
