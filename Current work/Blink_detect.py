import cv2
import dlib
import numpy as np
import math
import pandas as pd
from scipy.spatial import distance as dist
import os

import time
from datetime import datetime


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('0.mp4')

data = dict()
eye = []
eye1 = []
i = 1

p = 0
frames_open = 0
frames_closed = 0
no_of_frames = 0
minn = 0
j = 0
cal_state = 1

while True:

    left_eye = []
    right_eye = []

    ret, frame = cap.read()
    if frame is None:
        break

    # Converting a color frame into a grayscale frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Creating an object in which we will sore detected faces
    faces = detector(frame)

    for face in faces:

        # Creating an object in which we will sore detected facial landmarks
        landmarks = predictor(frame, face)

        # RIGHT EYE COORDINATES DETECTING
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append([x, y])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # LEFT EYE COORDINATES DETECTING
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append([x, y])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # LEFT EYE EAR
        A1 = dist.euclidean(left_eye[1], left_eye[5])
        B1 = dist.euclidean(left_eye[2], left_eye[4])
        distance_left = (A1 + B1) / 2

        # DISTANCE
        distance = round(distance_left, 2)
        eye.append(distance)
        eye1.append(distance)
        # print(distance)

        no_of_frames = no_of_frames + 1

        # if cal_state == 1:
        eye.sort()
        minn = eye[:1]
        # print(minn)

        eye1 = [x - minn for x in eye1]

        # for z in range(len(eye1)):
        #     print(eye1[z])

        maxi = 0
        for i in eye1:
            if i > maxi:
                maxi = i

        for distance in eye1:
            if distance <= (maxi * 0.2):
                frames_closed = frames_closed + 1
            else:
                frames_open = frames_open + 1

        print("closed ", frames_closed)
        print("open ", frames_open)

        # P E R C L O S VALUE
        p = (frames_closed / no_of_frames) * 100
        print('Percentage of p', p)
        # print("MAXIMUM",maxi)
        cv2.putText(frame, "PERCLOS: {:.2f}".format(p), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('video', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
