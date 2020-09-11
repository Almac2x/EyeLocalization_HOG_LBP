# Website From: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# Website to Reference: https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import time
import numpy as np
import csv
from Descriptors.HOG import HOG
from Descriptors.localbinarypatterns import LocalBinaryPatterns


class LBP_HOG:
    def __init__(self, image):
        self.image = image
        self.Get_LBP = LocalBinaryPatterns()

    def getLBPHOG(self, image, file_name, frame):
        with open(file_name+'_Descriptor_RunTime_LBPHOG.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            start_time = time.time()

            HOG_hist = HOG.getHOGimage(image, file_name, frame)
            LBP_hist = self.Get_LBP.describe(image, file_name, frame)  # get the LBP histogram here.

            feat = np.hstack([LBP_hist, HOG_hist])
            elapse_time = (time.time() - start_time)
            writer.writerow([frame, elapse_time])
            print("--- %s seconds to convert LBP_HOG ---" % elapse_time)

        return feat


