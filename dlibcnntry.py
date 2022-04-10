# # OpenCV's face detection network
# opencv_fd:
#   load_info:
#     url: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
#     sha1: "15aa726b4d46d9f023526d85537db81cbc8dd566"
#   model: "opencv_face_detector.caffemodel"
#   config: "opencv_face_detector.prototxt"
#   mean: [104, 177, 123]
#   scale: 1.0
#   width: 300
#   height: 300
#   rgb: false
#   sample: "object_detection"

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

from typing import Tuple
import dlib

from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta

now = datetime.now()
print(now)
start_time = now.strftime("%H:%M:%S")
print("[INFO] Current Time =", start_time)

start = timer()

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())
# [TODO] Make sure can run atleast once
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cap = cv2.VideoCapture('blink_video.mp4')
vid_name = 'S08_MD_3.mov'

cap = cv2.VideoCapture(vid_name)
print(vid_name)
# start of video frame (change based on last run, frame_n_end)
# if first run set as 1
# 3/ 15/ 28
# 5400/ 27000/ 50400
frame_n_start = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n_start - 1)
frame_n_end = frame_n_start
# can set the desired window size here
# please leave some additional space of at least 30 pixels for height so that we can display
# frame number and other parameters
desired_window_size: Tuple[int, int] = (220, 220)  # (width, height)

p = 0
frames_open = 0
frames_closed = 0
no_of_frames = 0
minn = 0
j = 0
break_flag = 0
cal_state = 1
no_face = 0
max_h = 0
max_w = 0
min_h = 1000
min_w = 1000
minute_counter = 0

old_startX, old_endX, old_startY, old_endY = 0, 0, 0, 0

# Set this based on how long you want to run the program
# Default is set as 300 frames (10 seconds)
length_of_run = 30


# function to pad the ROI based on desired window size
def pad_vframes(vframes, desired_size):
    desired_size_w = int(desired_size[0])
    desired_size_h = int(desired_size[1])
    old_size = vframes.shape[:2]  # old_size is in (height, width) format

    delta_w = desired_size_w - int(old_size[1])
    delta_h = desired_size_h - int(old_size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    vframes_padded = cv2.copyMakeBorder(vframes, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)
    return vframes_padded


# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    # frame = vs.read()
    # frame = imutils.resize(frame, width=400)

    left_eye = []
    right_eye = []

    if break_flag == 1:
        break

    ret, frame = cap.read()
    if ret:
        no_of_frames += 1
        frame = imutils.resize(frame, width=400)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        if detections is not None:
            print(detections.size)
            # print("got faces")
            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < 0.5:
                    continue
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                print(box)
                (startX, startY, endX, endY) = box.astype("int")

                # [TODO] FROM HERE ONLY NEED TO CROP THE FACE AND CAN START DETECT EYES

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        else:
            print("no face")
            no_face += 1
            startX, endX, startY, endY = old_startX, old_endX, old_startY, old_endY

        # roi_color = frame[startY:endY, startX:endX]
        # show the output frame
        cv2.imshow("Frame", frame)
        # time.sleep(5)
        # To determine the max window size
        # if len(roi_color[0]) > max_h:
        #     max_h = len(roi_color[0])
        #
        # if len(roi_color[1]) > max_w:
        #     max_w = len(roi_color[1])
        #
        # if len(roi_color[0]) < min_h:
        #     min_h = len(roi_color[0])
        #
        # if len(roi_color[1]) < min_w:
        #     min_w = len(roi_color[1])

        # this function we feed in the cropped frame and desired window size
        # output = pad_vframes(roi_color, desired_window_size)

        frame_n_end += 1

        old_startX, old_endX, old_startY, old_endY = startX, endX, startY, endY

        # this part is to stop the code when the letter 'q' is pressed on keyboard (only if cv2.imshow is used)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break_flag = 1
            end = timer()
            print(timedelta(seconds=end - start))
            break

        if frame_n_end == (frame_n_start + length_of_run):
            break_flag = 1
            end = timer()
            print('[INFO] processing time =', timedelta(seconds=end - start))
            break

    else:

        end = timer()
        print('[INFO] processing time =', timedelta(seconds=end - start))
        break

    if (no_of_frames % 1800) == 0:
        minute_counter += 1
        print('[INFO] Minutes completed =', minute_counter)
        print('[INFO] Number of frames completed =', no_of_frames)

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
print('[INFO] number of frames =', no_of_frames)
print('[INFO] frames start =', frame_n_start)
print('[INFO] frames end =', frame_n_end)
print('[INFO] number of no_face =', no_face)
print('[INFO] max_h =', max_h)
print('[INFO] max_w =', max_w)
print('[INFO] min_h =', min_h)
print('[INFO] min_w =', min_w)
print("done")
