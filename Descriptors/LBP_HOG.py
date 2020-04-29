# Website From: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# Website to Reference: https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import time
import numpy as np


class LBP_HOG:

    @staticmethod
    @staticmethod
    def __init__(image):
        IMG = LBP_HOG.getLBPHOG(image)

    def getLBPHOG(image_to_convert):
        start_time = time.time()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block, visualize, normalize)  # HOG descriptor here.
        hist = desc.describe(gray)  # get the LBP histogram here.

        # print(hist)
        print("--- %s seconds to convert HOG ---" % (time.time() - start_time))

        return hist

