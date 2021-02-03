# import the necessary packages
from skimage import feature
import numpy as np
import time
import csv


class LBP:
    def __init__(self):
        # store the number of points and radiusx
        self.numPoints = 8
        self.radius = 2

    def describe(self, image, file_name, frame):
        # Initializes Time to show how long it computes
        with open(file_name + '_Descriptor_RunTime_LBP.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            Start_Time = time.time()
            # compute the Local Binary Pattern representation
            # of the image, and then use the LBP representation
            # to build the histogram of patterns
            lbp = feature.local_binary_pattern(image, P=8, R=2, method="uniform")
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))

            # Initializes Time to show how long it computes
            Time_Compute = time.time() - Start_Time
            print("--- {} seconds to convert LBP ---".format(Time_Compute))
        # return the histogram of Local Binary Patterns
        return hist
