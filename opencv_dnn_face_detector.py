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

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cap = cv2.VideoCapture('blink_video.mp4')
vid_name = 'S03_MD.mp4'

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

# roi_buffer = 250
desired_window_size: Tuple[int, int] = (220, 220)  # (width, height)


p = 0
frames_open = 0
frames_closed = 0
no_of_frames = 0
minn = 0
j = 0
break_flag = 0
cal_state = 1  # why do i have this?
no_face = 0
max_h = 0
max_w = 0
min_h = 1000
min_w = 1000
minute_counter = 0

old_startX, old_endX, old_startY, old_endY = 688, 791, 273, 377

left_eye = []
right_eye = []
ear_list = []
ear_avglist = []
eyes_signal = []
blink_signal = []
perclos_list = [0]*1800
blink_dur = 0
blink_counter = 0
blink_update = 0
eye_close_sum = 0

# Set this based on how long you want to run the program
# Default is set as 300 frames (10 seconds)
length_of_run = 300


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


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# Defining the Euclidean distance
def euclidean_distance(leftx, lefty, rightx, righty):
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


# Defining the eye aspect ratio
def get_ear(eye_points, facial_landmarks, frame_ge, drawline=False):
    # Defining the left point of the eye
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    # Defining the right point of the eye
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
    # Defining the top mid-point of the eye
    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    # Defining the bottom mid-point of the eye
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    if drawline:
        # Drawing horizontal and vertical line
        hor_line = cv2.line(frame_ge, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0),
                            1)
        ver_line = cv2.line(frame_ge, (center_top[0], center_top[1]), (center_bottom[0], center_bottom[1]),
                            (255, 0, 0), 1)
    # Calculating length of the horizontal and vertical line
    hor_line_length = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_length = euclidean_distance(center_top[0], center_top[1], center_bottom[0],
                                         center_bottom[1])
    # Calculating eye aspect ratio
    ear = (ver_line_length / hor_line_length)*100

    return ear, left_point  # include right point later

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels

    if break_flag == 1:
        break

    ret, frame = cap.read()
    clone_img = frame.copy()
    (clone_h, clone_w) = clone_img.shape[:2]
    # print(clone_h, clone_w)
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

        # [TODO] DOES THIS CONDITION ACTUALLY CHECK IF THERE ARE FACES OR WOULD CONFIDENCE BE A BETTER CONDITION?
        if detections is not None:
            # print(detections.size)
            # print("got faces")

            # loop over the detections in 1 frame (if got more than 1 face)
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
                box = detections[0, 0, i, 3:7] * np.array([clone_w, clone_h, clone_w, clone_h])
                box = box.astype("int")
                (startX, startY, endX, endY) = box

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10

                roi_color = clone_img[startY:endY, startX:endX]
                roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

                landmarks = predictor(roi_gray, dlib.rectangle(startX-startX, startY-startY, endX-startX, endY-startY))

                left_eye_ratio, left_eye_left_point = get_ear([36, 37, 38, 39, 40, 41], landmarks, frame)
                # Calculating right eye aspect ratio
                right_eye_ratio, right_eye_left_point = get_ear([42, 43, 44, 45, 46, 47], landmarks, frame)
                # Calculating aspect ratio for both eyes
                ear_avg = (left_eye_ratio + right_eye_ratio) / 2
                # Rounding ear_avg on two decimal places
                # blinking_ratio_1 = ear_avg * 100
                ear_rounded = np.round(ear_avg)
                # blinking_ratio_rounded = ear_rounded / 100
                # Appending blinking ratio to a list eye_blink_signal
                ear_list.append(ear_rounded)
                ear_avglist.append(ear_rounded)
                if len(ear_avglist) > 3:
                    ear_avglist.pop(0)

                ear_plot = sum(ear_avglist)/len(ear_avglist)
                # [TODO] FIND SUITABLE THRESHOLD TO DETECT CLOSED EYES
                if ear_avg < 18:
                    if eyes_state == 0:
                        blink_update = 1
                        blink_start = no_of_frames
                    eyes_state = 1  # close
                    blink_dur += 1

                else:
                    eyes_state = 0  # open
                    if blink_update == 1:
                        blink_counter += 1
                        blink_end = no_of_frames - 1
                        # keeps track of blink events
                        blink_signal.append((blink_start, blink_end, blink_dur))
                        blink_update = 0
                    blink_dur = 0
                eyes_signal.append(eyes_state)

                perclos = (eye_close_sum/1800)*100
                popvalue = perclos_list.pop(0)
                perclos_list.append(eyes_state)
                eye_close_sum = eye_close_sum + eyes_state - popvalue

                # RIGHT EYE COORDINATES DETECTING
                for n in range(42, 48):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    right_eye.append([x, y])
                    cv2.circle(roi_color, (x, y), 1, (0, 0, 255), -1)

                # LEFT EYE COORDINATES DETECTING
                for n in range(36, 42):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    left_eye.append([x, y])
                    cv2.circle(roi_color, (x, y), 1, (0, 0, 255), -1)

        else:
            print("no face")
            no_face += 1
            startX, endX, startY, endY = old_startX, old_endX, old_startY, old_endY


        # show the frame bounding box
        cv2.imshow("Frame", frame)
        # cv2.imshow("clone", clone_img)
        # time.sleep(5)
        # To determine the max window size
        if len(roi_color[0]) > max_h:
            max_h = len(roi_color[0])

        if len(roi_color[1]) > max_w:
            max_w = len(roi_color[1])

        if len(roi_color[0]) < min_h:
            min_h = len(roi_color[0])

        if len(roi_color[1]) < min_w:
            min_w = len(roi_color[1])

        # this function we feed in the cropped image and desired window size
        output = pad_vframes(roi_color, desired_window_size)
        image_scaled = imutils.resize(roi_color, width=220)

        cv2.imshow("image", image_scaled)

        # show the frame bounding box
        cv2.imshow("Output", output)

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
