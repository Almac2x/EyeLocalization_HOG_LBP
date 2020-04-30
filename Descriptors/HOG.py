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

        H = hog(image_to_convert, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", feature_vector=False)
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #
        # ax1.axis('off')
        # ax1.imshow(image_to_convert, cmap=plt.cm.gray)
        # ax1.set_title('Input image')
        #
        # # Rescale histogram for better display
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        #
        # ax2.axis('off')
        # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        # ax2.set_title('Histogram of Oriented Gradients')
        # plt.show()
        (hist, _) = np.histogram(H)
        # normalize the histogram
        eps = 1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # print(hist)
        print("--- %s seconds to convert HOG ---" % (time.time() - start_time))

        return hist
