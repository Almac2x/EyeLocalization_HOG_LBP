#Website From: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
# Website From: https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
import cv2, os
import numpy as np
import time
class LBP:


    @staticmethod
    def __init__(image):

        IMG = LBP.getLBPimage(image)

    def getLBPimage(image):
        start_time = time.time()
        '''
        == Input ==
        gray_image  : color image of shape (height, width)

        == Output ==
        imgLBP : LBP converted image of the same shape as
        '''

        ### Step 0: Step 0: Convert an image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgLBP = np.zeros_like(gray_image)
        neighboor = 3
        for ih in range(0, image.shape[0] - neighboor):
            for iw in range(0, image.shape[1] - neighboor):
                ### Step 1: 3 by 3 pixel
                img = gray_image[ih:ih + neighboor, iw:iw + neighboor]
                center = img[1, 1]
                img01 = (img >= center) * 1.0
                img01_vector = img01.T.flatten()
                # it is ok to order counterclock manner
                # img01_vector = img01.flatten()
                ### Step 2: **Binary operation**:
                img01_vector = np.delete(img01_vector, 4)
                ### Step 3: Decimal: Convert the binary operated values to a digit.
                where_img01_vector = np.where(img01_vector)[0]
                if len(where_img01_vector) >= 1:
                    num = np.sum(2 ** where_img01_vector)
                else:
                    num = 0
                imgLBP[ih + 1, iw + 1] = num

        print("--- %s seconds to convert LBP ---" % (time.time() - start_time))
        return (imgLBP)