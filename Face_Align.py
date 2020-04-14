#Website From: https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/
import cv2
import time

#Inializes Haar Cascade Eye Classifier
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")



#Loads the Image
img_path = "images/adrian.jpg"
img = cv2.imread(img_path)
#Copies the Image
img_raw = img.copy()

#Gets image from Video Stream
cap = cv2.VideoCapture(0)

# start the FPS throughput estimator
Start_TIme = time.time()
Frame_Counter = 0
Tick  = 0
FPS = 0


while (True):
    ret, frame = cap.read()
    # Converts the Image to Greyscale
    gray_img = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

    # Eye Detection
    eyes = eye_detector.detectMultiScale(gray_img)

    index = 0
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)

        cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0,0,0), 2)
        index = index + 1

    cv2.putText(frame, "FPS: " + str(FPS), (85, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Calculates the FPS in Realtime
    Frame_Counter = Frame_Counter + 1
    Time_Now = time.time() - Start_TIme
    if (Time_Now - Tick) >= 1:
        Tick = Tick + 1
        FPS = Frame_Counter
        Frame_Counter = 0





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow("Blando",frame)

#Shows the image

cap.release()
cv2.destroyAllWindows()


