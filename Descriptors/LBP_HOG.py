import time
import numpy as np
import csv
from Descriptors.HOG import HOG
from Descriptors.LBP import LBP


class LBP_HOG:
    def __init__(self, image):
        self.image = image
        self.Get_LBP = LBP()

    def getLBPHOG(self, image, file_name, frame):
        with open(file_name+'_Descriptor_RunTime_LBPHOG.csv', 'a', newline='') as file:
            writer = csv.writer(file)

            start_time = time.time()

            HOG_hist = HOG.getHOGimage(image, file_name, frame)
            LBP_hist = self.Get_LBP.describe(image, file_name, frame)  # get the LBP histogram here.

            feat = np.hstack([LBP_hist, HOG_hist])
            elapse_time = (time.time() - start_time)
            print("--- %s seconds to convert LBP_HOG ---" % elapse_time)

        return feat


