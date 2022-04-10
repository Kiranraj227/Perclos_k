import shutil
import numpy as np
import cv2
from typing import Tuple
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import os
import statistics
import mediapipe as mp
import time
import pandas as pd
import logging
import dlib
from imutils.video import VideoStream
import argparse
import imutils


# def rect_to_bb(rect):
#     # take a bounding predicted by dlib and convert it
#     # to the format (x, y, w, h) as we would normally do
#     # with OpenCV
#     bb_x = rect.left()
#     bb_y = rect.top()
#     bb_w = rect.right() - bb_x
#     bb_h = rect.bottom() - bb_y
#
#     # return a tuple of (x, y, w, h)
#     return bb_x, bb_y, bb_w, bb_h


def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    bb_start_x = rect.left()
    bb_start_y = rect.top()
    bb_end_x = rect.right()
    bb_end_y = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    bb_start_x = max(0, bb_start_x)
    bb_start_y = max(0, bb_start_y)
    bb_end_x = min(bb_end_x, image.shape[1])
    bb_end_y = min(bb_end_y, image.shape[0])
    # compute the width and height of the bounding box
    bb_w = bb_end_x - bb_start_x
    bb_h = bb_end_y - bb_start_y
    # return our bounding box coordinates
    return bb_start_x, bb_start_y, bb_w, bb_h


dlib_path = 'E:\\Perclos_k_io\\Dependant files\\Dlib\\shape_predictor_68_face_landmarks.dat'
dlib_face_detector_path = 'E:\\Perclos_k_io\\Dependant files\\Dlib\\mmod_human_face_detector.dat'

# Initializing the face detector and facial landmark predictor
print("[INFO] loading models...")
detector = dlib.cnn_face_detection_model_v1(dlib_face_detector_path)
predictor = dlib.shape_predictor(dlib_path)
cap = cv2.VideoCapture('E:\\Perclos_k_io\\test_video.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 160)
frame_counter = 0

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels

    ret, frame = cap.read()
    # frame = frame[bb_start[1]:bb_end[1], bb_start[0]:bb_end[0]]

    if ret:
        fps_start = time.time()

        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        # wanna use this?
        # frame.flags.writeable = False

        # frame = cv2.resize(frame, (600, 600)) # is this necessary

        (h, w) = frame.shape[:2]

        # convert to rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector(frame_rgb)
        print('process results')
        # Draw the face mesh annotations on the frame.
        # frame.flags.writeable = True

        #  Conversion return value dlib The rectangular bounding box is opencv Rectangular bounding box , And make sure it falls in the image
        # loop over the face detections

        frame_counter += 1
        print(frame_counter)

        # print(results[0].confidence)
        # print(results[0].rect)
        # confidence_level = results[0].confidence
        try:

            for r in results:
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = convert_and_trim_bb(frame, r.rect)
                print('in loop')

                (startX, startY, endX, endY) = x, y, x + w, y + h

                roi_color = frame[startY:endY, startX:endX]
                # roi_color = clone_img
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

                landmarks = predictor(roi_gray,
                                      dlib.rectangle(startX - startX, startY - startY, endX - startX,
                                                     endY - startY))

                # Draw red bounding box of face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                confidence_level = r.confidence
                # show the face number
                cv2.putText(frame, "Confidence: {}".format(confidence_level), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
        except:
            print('no face')

        cv2.imshow("Output", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print('user ended it')
            cap.release()
            cv2.destroyAllWindows()
            break

cap.release()
cv2.destroyAllWindows()
print('done')
