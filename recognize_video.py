# USAGE
# python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from Eye_Detection import Eyes

# Import Feature Descriptors
from pyimagesearch.nms import non_max_suppression_fast

#Loads Eye Detector
Eye_Detector = Eyes("HOG")

def getEyes(roi):
    #Gets eyes location into an array
    Eyes = Eye_Detector.getEyes(roi)

    # Performs NMS to approximate boxes
    nms = non_max_suppression_fast(Eyes, 0.3)
    return nms




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
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

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
input_video='test_v.mp4'
vs = VideoStream(input_video).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
Start_TIme = time.time()
Frame_Counter = 0
Tick = 0
startY = 0
startX = 0
endY = 0
endX = 0
#Change here the descriptors use
Descriptor = "HOG"




# loop over frames from the video file stream
while True:
    found_face = False
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            # found_face = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.putText(frame, "FPS: " + str(FPS), (startX, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # Crops the Face
            roi = frame[startY:int(endY), startX:endX]
            #Resizes Shape
            roi_resize = cv2.resize(roi, (192, 192), interpolation=cv2.INTER_AREA)

            if (found_face == True):
                # Computes Eye Locations
                if Descriptor == "LBP":
                    Eyes = Eye_Detector.getEyes(cv2.imread("dataset/Aptina/Eye/s0012_01282_0_0_0_0_1_03.png"))
                elif Descriptor == "HOG":
                    Eyes = Eye_Detector.getEyes(roi_resize)

                # Draws the boxes for eyes
                nms = non_max_suppression_fast(Eyes, 0.3)

                for (startX, startY, endX, endY) in nms:
                    cv2.rectangle(roi, (startX, startY), (endX, endY), (255, 0, 0), 2)


    # update the FPS counter
    fps.update()
    # Calculates the FPS in Realtime
    Frame_Counter = Frame_Counter + 1
    Time_Now = time.time() - Start_TIme
    if (Time_Now - Tick) >= 1:
        Tick = Tick + 1
        FPS = Frame_Counter
        Frame_Counter = 0



    # show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
