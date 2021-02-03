# import the necessary packages
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

args_detector = "Face_Detection_Model"
args_embedding_model = "Process/openface_nn4.small2.v1.t7"
args_recognizer = "Process/recognizer.pickle"
args_le = "Process/le.pickle"

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


# Change here the descriptors use
Descriptor = "LBP_HOG"


video_file = 'Test_Videos/RealTime.mp4'

# Loads Eye Detector
Eye_Detector = Eyes(Descriptor)
print("[INFO] starting video stream...")

cap = cv2.VideoCapture(0)
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
filename = video_file.split('/')
file = filename[1].split('.')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
filepath = 'Output_Eye_Detection/' + Descriptor + '/' + filename[1]
out = cv2.VideoWriter(filepath, fourcc, 30.0, (1280, 720))
frame_count = 1
# loop over frames from the video file stream
while cap.isOpened():

    # grab the frame from the threaded video stream
    ret, frame = cap.read()
    Start_FrameTime = time.time()
    # resize the frame to have a width of 720 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    if ret == True:
        frame = imutils.resize(frame, width=720)
        (h, w) = frame.shape[:2]
        found_face = False
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
                found_face = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                cv2.putText(frame, Descriptor, (5, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # Crops the Face to reduce the number of regions to be processed by the eye localization
                roi = frame[startY:(startY+(endY//2)), startX:endX]
                # Resizes Shape
                roi_resize = cv2.resize(roi, (192, 192), interpolation=cv2.INTER_AREA)

                if found_face == True:
                    # Computes Eye Locations
                    if Descriptor == "LBP":
                        Eyes = Eye_Detector.getEyes(roi, file[0], frame_count)
                    elif Descriptor == "HOG":
                        Eyes = Eye_Detector.getEyes(roi, file[0], frame_count)
                    elif Descriptor == "LBP_HOG":
                        Eyes = Eye_Detector.getEyes(roi, file[0], frame_count)

                    # Uses Non Max Suppression to average overlapping boxes
                    nms = non_max_suppression_fast(Eyes, 0.003)

                    # Writes each box to the region of interest (roi)
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

        # show the Process frame
        frame = imutils.resize(frame, width=1280, height=720)
        cv2.imshow("Frame", frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            print(frame.shape)
            break
        frame_count += 1

    else:
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
out.release()





