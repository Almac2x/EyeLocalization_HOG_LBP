from skimage.feature import hog
import time
import numpy as np


class HOG:

    @staticmethod
    def __init__(image):
        IMG = image

    def getHOGimage(image_to_convert):
        start_time = time.time()

        # fd = hog(image_to_convert, orientations=9, pixels_per_cell=(8, 8),
        #                   cells_per_block=(2, 2), block_norm="L2", feature_vector=True)

        H = hog(image_to_convert, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", feature_vector=False)

        (hist, _) = np.histogram(H)
        # normalize the histogram
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # print(hist)
        print("--- %s seconds to convert HOG ---" % (time.time() - start_time))

        return hist
