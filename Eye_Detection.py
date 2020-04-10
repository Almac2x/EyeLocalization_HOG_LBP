import cv2,pickle,argparse,time
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
    Model_Path = "Eye_Detection_Model/finalized_model.sav"
    loaded_model = pickle.load(open(Model_Path, 'rb'))

    #Converts to LocalBinaryPatter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    hist = desc.describe(gray)

    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW


            prediction = loaded_model.predict(hist.reshape(1, -1))

            # display the image and the prediction
            cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)




