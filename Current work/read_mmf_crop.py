# Libraries
import mne
import matplotlib.pyplot as plt
import scipy.io
from timeit import default_timer as timer
from datetime import timedelta
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

import time
from datetime import datetime

now = datetime.now()
print(now)
start_time = now.strftime("%H:%M:%S")
print("Current Time =", start_time)

start = timer()

'''
fname = "S06_20170817_031757.mff"
raw = mne.io.read_raw_egi(fname)
print(raw.ch_names)
'''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('S06_20170817_032034.mov')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output_2.mp4', fourcc, 30, (240, 240))

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
break_flag = 0
cal_state = 1


def pad_vframes(vframes, desired_size):
    desired_size_w = int(desired_size[1])
    desired_size_h = int(desired_size[0])
    old_size = vframes.shape[:2]  # old_size is in (height, width) format

    delta_w = desired_size_w - int(old_size[1])
    delta_h = desired_size_h - int(old_size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    vframes_padded = cv2.copyMakeBorder(vframes, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)
    return vframes_padded


while True:

    left_eye = []
    right_eye = []

    if break_flag == 1:
        break

    ret, frame = cap.read()
    if ret:
        no_of_frames += 1
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(fps_cap, w, h)

        # Converting a color frame into a grayscale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Creating an object in which we will sore detected faces
        faces = detector(gray)

        if faces:
            # print("got face")
            # cropping the roi of face
            for face in faces:
                x_start, y_start = face.left(), face.top()
                x_end, y_end = face.right(), face.bottom()

        else:
            print("no face")
            x_start, x_end, y_start, y_end = old_x_start, old_x_end, old_y_start, old_y_end

        # print("Coordinates here---", x_start, x_end, y_start, y_end)
        roi_gray = gray[(y_start - 10):(y_end + 10), (x_start - 10):(x_end + 10)]
        roi_color = frame[(y_start - 10):(y_end + 10), (x_start - 10):(x_end + 10)]

        # Creating an object in which we will store detected facial landmarks
        """
        landmarks = predictor(roi_gray, face)

        # RIGHT EYE COORDINATES DETECTING
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append([x, y])
            cv2.circle(roi_color, (x, y), 4, (0, 0, 255), -1)

        # LEFT EYE COORDINATES DETECTING
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append([x, y])
            cv2.circle(roi_color, (x, y), 4, (0, 0, 255), -1)
        """
        cv2.putText(roi_color, "PERCLOS: {:.2f}".format(p), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imshow('video', roi_color)
        output = pad_vframes(roi_color, (240, 240))
        # output = np.zeros((int(h - 500.0), int(w - 1060.0), 3), dtype="uint8")
        print(output.shape)
        # output[10:(y_end - y_start + 30), 10:(x_end - x_start + 30)] = roi_color
        cv2.putText(output, "Frame: {}".format(no_of_frames), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.imshow('video_crop', output)

        out.write(output)
        old_x_start, old_x_end, old_y_start, old_y_end = x_start, x_end, y_start, y_end
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break_flag = 1
            end = timer()
            print(timedelta(seconds=end - start))
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("done")
