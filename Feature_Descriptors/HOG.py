import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import time
class HOG:

    @staticmethod
    def __init__(image):
        IMG = HOG.getLBPimage(image)

    def getHOGimage(image_to_convert):
        start_time = time.time()

        fd,hog_image = hog(image_to_convert, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        print("--- %s seconds to convert HOG ---" % (time.time() - start_time))
        return (hog_image_rescaled)