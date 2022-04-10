# notes:
# Currently (12/11/2021), the processing time takes about 4x of video length
# 10s video will take 40s to process

# Libraries
from timeit import default_timer as timer
from datetime import timedelta
from typing import Tuple

import cv2
import dlib
import numpy as np

from datetime import datetime

now = datetime.now()
print(now)
start_time = now.strftime("%H:%M:%S")
print("[INFO] Current Time =", start_time)

start = timer()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# cap = cv2.VideoCapture('blink_video.mp4')

cap = cv2.VideoCapture('S01_MD.mov')

# start of video frame (change based on last run, frame_n_end)
# if first run set as 1
frame_n_start = 36001
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


# Set this based on how long you want to run the program
# Default is set as 300 frames (10 seconds)
length_of_run = 54000

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
            # print("no face")
            no_face += 1
            x_start, x_end, y_start, y_end = old_x_start, old_x_end, old_y_start, old_y_end

        # print("Coordinates here---", x_start, x_end, y_start, y_end)
        roi_gray = gray[y_start:y_end, x_start:x_end]
        roi_color = frame[y_start:y_end, x_start:x_end]

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
        # cv2.imshow('video', roi_color)

        if len(roi_color[0]) > max_h:
            max_h = len(roi_color[0])

        if len(roi_color[1]) > max_w:
            max_w = len(roi_color[1])

        if len(roi_color[0]) < min_h:
            min_h = len(roi_color[0])

        if len(roi_color[1]) < min_w:
            min_w = len(roi_color[1])

        # this function we feed in the cropped frame and desired window size
        output = pad_vframes(roi_color, desired_window_size)

        # cv2.putText(output, "PERCLOS: {:.2f}".format(p), (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(output, "Frame: {}".format(no_of_frames), (10, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.imshow('video_crop_1', output)
        # cv2.putText(output, "Frame: {}".format(frame_n_end), (10, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.imshow('video', output)
        frame_n_end += 1

        old_x_start, old_x_end, old_y_start, old_y_end = x_start, x_end, y_start, y_end

        # This lines of code is to stop the crop up to a certain number of frames based on length_of_run
        if frame_n_end == (frame_n_start + length_of_run):
            break_flag = 1
            end = timer()
            print('[INFO] processing time =', timedelta(seconds=end - start))
            break

        # this part is to stop the code when the letter 'q' is pressed on keyboard (only if cv2.imshow is used)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break_flag = 1
            end = timer()
            print(timedelta(seconds=end - start))
            break

        if (no_of_frames % 1800) == 0:
            minute_counter += 1
            print('[INFO] Minutes completed =', minute_counter)
            print('[INFO] Number of frames completed =', no_of_frames)
    else:

        end = timer()
        print('[INFO] processing time =', timedelta(seconds=end - start))
        break
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
