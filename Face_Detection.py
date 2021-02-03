# import the necessary packages
import os
import pickle

import cv2
import numpy as np


class Face_Detection:
    def __init__(self):
        # store the number of points and radiusx
        self.numPoints = 8
        self.radius = 2

        args_detector = "Face_Detection_Model"
        args_embedding_model = "Process/openface_nn4.small2.v1.t7"
        args_recognizer = "Process/recognizer.pickle"
        args_le = "Process/le.pickle"
        self.args_confidence = .9

        print("[INFO] loading face detector...")
        protoPath = r"Face_Detection_Model/deploy.prototxt"
        modelPath = os.path.sep.join([args_detector,
                                      "res10_300x300_ssd_iter_140000.caffemodel"])

        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load our serialized face embedding model from disk

        self.embedder = cv2.dnn.readNetFromTorch(args_embedding_model)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(args_recognizer, "rb").read())
        self.le = pickle.loads(open(args_le, "rb").read())
        print("[INFO] loading face recognizer...")


    def getFace(self,Image):

        #Face ARRAY
        Faces = []
        box = []
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions
        # image = imutils.resize(Image, width=600)
        (h, w) = Image.shape[:2]  # Aspect Ratio

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(Image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > self.args_confidence:
                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                Faces.append(box)

        return box


