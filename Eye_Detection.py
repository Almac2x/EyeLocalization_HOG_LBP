import cv2, pickle
from Descriptors.LBP_HOG import LBP_HOG
from Descriptors.LBP import LBP
from pyimagesearch.helpers import sliding_window
from Descriptors.HOG import HOG
import numpy as np
import csv
import time

class Eyes:
    def __init__(self, Descriptor):
        # Chooses what descriptor to use
        self.Descriptor = Descriptor
        if Descriptor == "LBP":
            self.Model_Path = "Eye_Detection_Model/LBP_3_KFOLD_0.8426 _KF#2.sav"
        elif Descriptor == "HOG":
            self.Model_Path = "Eye_Detection_Model/HOG_3_KFOLD_0.778 _KF#2.sav"
        elif Descriptor == "LBP_HOG":
            self.Model_Path = "Eye_Detection_Model/LBPHOG_3_KFOLD_0.852 _KF#1.sav"

    def getEyes(self, image, file_name, frame):
        # Loads the model to be used
        loaded_model = pickle.load(open(self.Model_Path, 'rb'))
        # Starts of Eye Detection

        # Initializes box array to store eye locations
        Eye_Box_Loc = []

        # load the image and define the window width and height
        (winW, winH) = (64, 64)

        if self.Descriptor == "LBP":
            # initialize the local binary patterns descriptor along with
            desc = LBP()
        elif self.Descriptor == "LBP_HOG":
            desc = LBP_HOG("Run")

        # Converts to BGR2GRAY
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # loop over the image pyramid
        # for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        frame_number = 1
        with open(file_name+'_SlidingWindow_RunTime.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            Start_Time = time.time()
            for (x, y, window) in sliding_window(image, stepSize=16, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                clone = image.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                crop_img = gray[y:y + winH, x:x + winW]
                Box = [x, y, x + winW, y + winH]

                # Describes the image
                if (self.Descriptor == "LBP"):
                    hist = desc.describe(crop_img, file_name, frame)
                elif (self.Descriptor == "HOG"):
                    hist = HOG.getHOGimage(crop_img, file_name, frame)
                elif self.Descriptor == "LBP_HOG":
                    hist = desc.getLBPHOG(crop_img, file_name, frame)

                # Loads Prediction Model
                reshape_Desc = hist.reshape(1, -1)
                prediction = loaded_model.predict(reshape_Desc)
                # self.loaded_model.classes
                confidence_level = loaded_model.decision_function(reshape_Desc)

                if self.Descriptor == "LBP":
                    Eye_Open_Confidence_Level = confidence_level[0] * 100
                    if prediction[0] == "Eyes_Left" or prediction[0] == "Eyes_Right":
                        Eye_Box_Loc.append(Box)

                elif self.Descriptor == "HOG":
                    Eye_Open_Confidence_Level = confidence_level[0] * 100
                    if prediction[0] == "Eyes_Left" or prediction[0] == "Eyes_Right":
                        Eye_Box_Loc.append(Box)

                elif self.Descriptor == "LBP_HOG":
                    Eye_Open_Confidence_Level = confidence_level[0] * 100
                    if prediction[0] == "Eyes_Left" or prediction[0] == "Eyes_Right":
                        Eye_Box_Loc.append(Box)

                print(prediction[0])
                print(Eye_Open_Confidence_Level)
                frame_number += 1
            Elapse_Time = time.time() - Start_Time
            writer.writerow([frame_number, Elapse_Time])

        return np.array(Eye_Box_Loc)













