# Website https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import cv2, pickle, argparse, time
from Feature_Descriptors.LBP import LBP
from Feature_Descriptors.localbinarypatterns import LocalBinaryPatterns
from pyimagesearch.helpers import pyramid, sliding_window


def getEyes(image):
    # Starts of Eye Detection

    # load the image and define the window width and height
    (winW, winH) = (64, 64)

    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []

    # initialize the local binary patterns descriptor along with

    # load the model from disk
    Model_Path = "Eye_Detection_Model/LBP_Oversize_Datasets_0.8676 _KF#3.sav"
    loaded_model = pickle.load(open(Model_Path, 'rb'))

    # Converts to LocalBinaryPatter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            clone = resized.copy()

            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)

            crop_img = gray[y:y + winH, x:x + winW]
            crop_img = cv2.resize(crop_img, (64, 64), interpolation=cv2.INTER_AREA)

            hist = desc.describe(crop_img, "Frame")
            # Loads Prediction Model
            prediction = loaded_model.predict(hist.reshape(1, -1))

            confidence_level = loaded_model.decision_function(hist.reshape(1, -1))

            print(prediction[0])
            print("Confidence Level: {}".format(loaded_model.decision_function(hist.reshape(1, -1))))

            # display the image and the prediction
            # XY = (x, y + winH)
            # cv2.putText(image, prediction[0], XY, cv2.FONT_HERSHEY_SIMPLEX,
            #           1.0, (0, 0, 255), 3)

            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)

            cv2.waitKey(0)
            time.sleep(0.025)

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
