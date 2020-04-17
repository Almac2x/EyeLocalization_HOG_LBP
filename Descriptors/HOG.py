# Website From: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# Website to Reference: https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import time
import numpy as np


class HOG:

    @staticmethod
    def __init__(image):
        IMG = HOG.getLBPimage(image)

    def getHOGimage(image_to_convert):
        start_time = time.time()

        # fd = hog(image_to_convert, orientations=9, pixels_per_cell=(8, 8),
        #                   cells_per_block=(2, 2), block_norm="L2", feature_vector=True)

        fd = hog(image_to_convert, orientations=8, pixels_per_cell=(16, 16),
                  cells_per_block=(1, 1), feature_vector=True)

        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        print("--- %s seconds to convert HOG ---" % (time.time() - start_time))
        return (fd)
