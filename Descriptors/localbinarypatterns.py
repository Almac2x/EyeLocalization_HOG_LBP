# import the necessary packages
from skimage import feature
import numpy as np
import time


class LocalBinaryPatterns:
    def __init__(self):
        # store the number of points and radiusx
        self.numPoints = 8
        self.radius = 2

    def describe(self, image):
        # Initializes Time to show how long it computes
        Start_Time = time.time()
        eps = 1e-7
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))

        # Initializes Time to show how long it computes
        Time_Compute = time.time() - Start_Time
        print("--- {}s seconds to convert LBP ---".format(Time_Compute))
        # return the histogram of Local Binary Patterns
        return hist
