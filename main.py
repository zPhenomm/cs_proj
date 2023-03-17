import cv2
import numpy as np
from playsound import playsound

NIGHT = False

cap = cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
deathcounter = 0
alarmcounter = 0
flag = False
eye_ROI = []

while True:
    ret, frame = cap.read()    # get webcam image
    if NIGHT:
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = ()
    faces = ()

    # get ROI of face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # detect eyes in ROI of face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        if NIGHT:
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 7)
        else:
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 12)

    # if false detection of additional eye, the correct ones are the ones parallel to each other
    if len(eyes) > 2:
        for i in range(0, len(eyes)):
            for j in range(0, len(eyes)):
                if eyes[i][2] - eyes[j][1] != 0 and eyes[i][1] - eyes[j][1] < 10:
                    eyes = [eyes[i], eyes[j]]
                    break
            break

    # get ROI of eyes
    if np.any(faces):
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_ROI.append(roi_color[ey:ey + eh, ex:ex + ew])

    # detecton logic
    if np.any(faces) and not np.any(eyes):
        deathcounter += 1
        if alarmcounter > 10:  # if alarm has played for 2 seconds and eyes are still closed, continue alarm
            alarmcounter = 0
            flag = False
    if np.any(faces) and np.any(eyes):
        if deathcounter > 0:
            deathcounter -= 1
        if flag:
            deathcounter = 0
            if alarmcounter > 10:  # if alarm has played for 2 seconds and eyes are detected, reset alarm
                flag = False
                alarmcounter = 0
    if deathcounter > 10 or flag:
        alarmcounter += 1
        # tint output red
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.zeros_like(frame)
        frame[..., 2] = gray
        # play alarm
        if not flag:
            playsound("alarm.mp3", False)
            flag = True

    cv2.imshow('Driver Cam', frame)
    eye_ROI = []

    k = cv2.waitKey(200)  # 5fps
    if k == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()
