# USAGE Website From: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
# python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os, time, pickle
from pyimagesearch.nms import non_max_suppression_fast
from Eye_Detection import Eyes

# construct the argument parser and parse the arguments
from pyimagesearch.helpers import pyramid, sliding_window

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
                help="path to input image")
ap.add_argument("-d", "--detector", required=False,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=False,
                help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=False,
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=False,
                help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

args_detector = "face_detection_model"
args_embedding_model = "openface_nn4.small2.v1.t7"
args_recognizer = "output/recognizer.pickle"
args_le = "output/le.pickle"
args_image = "images/Pili_3.jpg"
#Change here the descriptors use
Descriptor = "HOG"

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args_detector, "deploy.prototxt"])
modelPath = os.path.sep.join([args_detector,
                              "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args_embedding_model)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args_recognizer, "rb").read())
le = pickle.loads(open(args_le, "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(args_image)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]  # Aspect Ratio

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

#Loads Eye Detector
Eye_Detector = Eyes(Descriptor)

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections
    if confidence > args["confidence"]:
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
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        # draw the bounding box of the face along with the associated
        # probability
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Crops the Face
roi = image[startY:int(endY), startX:endX]
roi_resize = cv2.resize(roi, (192, 192), interpolation=cv2.INTER_AREA)

# Computes Eye Locations
if(Descriptor == "LBP"):
    Eyes = Eye_Detector.getEyes(cv2.imread("dataset/Aptina/Eye/s0012_01282_0_0_0_0_1_03.png"))
elif(Descriptor == "HOG"):
    cv2.imshow("Nani",roi_resize)
    Eyes = Eye_Detector.getEyes(roi_resize)

# Draws the boxes for eyes
nms = non_max_suppression_fast(Eyes, 0.3)

# loop over the bounding boxes for each image and draw them
for (startX, startY, endX, endY) in nms:
    cv2.rectangle(roi, (startX, startY), (endX, endY), (0, 255, 0), 2)

# for (startX, startY, endX, endY) in nms:
#    sx = startX_roi + startX
#    sy = startY_roi + startY
#
#    ex = endx_roi- endX
#    ey = endY_roi - endY
#
#    cv2.rectangle(image, (sx, sy), (ex, ey), (0, 255, 0), 2)


# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
