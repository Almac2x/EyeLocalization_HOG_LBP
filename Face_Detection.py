# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os, time, pickle
from pyimagesearch.nms import non_max_suppression_fast
from Eye_Detection import Eyes

class Face_Detection:
    def __init__(self):
        # store the number of points and radiusx
        self.numPoints = 8
        self.radius = 2
        self.args_detector = "face_detection_model"
        self.args_embedding_model = "openface_nn4.small2.v1.t7"
        self.args_recognizer = "output/recognizer.pickle"
        self.args_le = "output/le.pickle"
        self.args_confidence = "90"

        print("[INFO] loading face detector...")
        self.protoPath = os.path.sep.join([self.args_detector, "deploy.prototxt"])
        self.modelPath = os.path.sep.join([self.args_detector,
                                      "res10_300x300_ssd_iter_140000.caffemodel"])

        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # load our serialized face embedding model from disk
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(self.args_embedding_model)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(self.args_recognizer, "rb").read())
        self.le = pickle.loads(open(self.args_le, "rb").read())


    def getFace(self,Image):

        Faces = []

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions

        image = imutils.resize(Image, width=600)
        (h, w) = image.shape[:2]  # Aspect Ratio

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
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
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                                 (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                # perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j]

                # draw the bounding box of the face along with the associated
                # probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


        return image


