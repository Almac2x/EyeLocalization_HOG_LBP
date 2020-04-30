# Website From: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# Website to Reference: https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import time
import numpy as np
from Descriptors.HOG import HOG
from Descriptors.localbinarypatterns import LocalBinaryPatterns


class LBP_HOG:
    def __init__(self, image):
        self.image = image
        self.Get_LBP = LocalBinaryPatterns(24, 8)

    def getLBPHOG(self, image):
        start_time = time.time()

        HOG_hist = HOG.getHOGimage(image)
        LBP_hist = self.Get_LBP.describe(image,"Frame")  # get the LBP histogram here.

        feat = np.hstack([LBP_hist,HOG_hist])
        # print(hist)
        print("--- %s seconds to convert HOG ---" % (time.time() - start_time))

        return feat


