# Code to generate PERCLOS Dataset

# import the necessary packages
import shutil

from imutils.video import VideoStream
import time
import argparse
import numpy as np
import imutils
import cv2
from typing import Tuple
import dlib
from datetime import datetime
from timeit import default_timer as timer
from datetime import timedelta
import os
import statistics
import mediapipe as mp
import time
import logging



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


# function to crop roi with a buffer of 300x300
def roi_crop_frames(frame_in, new_xy, new_xy2, desired_size):
    desired_size_w = int(desired_size[0])
    desired_size_h = int(desired_size[1])
    old_size = frame_in.shape[:2]  # old_size is in (height, width) format

    delta_w = desired_size_w - int(old_size[1])
    delta_h = desired_size_h - int(old_size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    bb_w = new_xy2[0] - new_xy[0]
    bb_h = new_xy2[1] - new_xy[1]

    roi_crop_xy = ((new_xy[0] - ((desired_size_w - bb_w)/2)), (new_xy[1] - ((desired_size_h - bb_h)/2)))

    return roi_crop_xy


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# Defining the Euclidean distance
def euclidean_distance(leftx, lefty, rightx, righty):
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


# Defining the eye aspect ratio
def get_ear(eye_points, facial_landmarks, frame_ge, drawline=True):
    # Defining the left point of the eye
    left_point = (facial_landmarks[eye_points[0]][0], facial_landmarks[eye_points[0]][1])
    # Defining the right point of the eye
    right_point = (facial_landmarks[eye_points[2]][0], facial_landmarks[eye_points[2]][1])
    # Defining the top mid-point of the eye
    center_top = (facial_landmarks[eye_points[1]][0], facial_landmarks[eye_points[1]][1])
    # Defining the bottom mid-point of the eye
    center_bottom = (facial_landmarks[eye_points[3]][0], facial_landmarks[eye_points[3]][1])

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
    ear = (ver_line_length / hor_line_length) * 100

    return ear, left_point  # include right point later

def check_make_folder(path, vid_path, remove=False):
    if not remove:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        else:
            print('output folder {} exists'.format(vid_path))
    else:
        """
        Force delete folder
        """
        if not os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)



# print some time info
now = datetime.now()
print(now)
start_time = now.strftime("%H:%M:%S")
print("[INFO] Current Time =", start_time)
start = timer()
start_pitstop = start

print("[INFO] starting video stream...")

# subject_list = [('03', (580, 150)), ('05', (580, 150)), ('08', (470, 200)), ('11', (470, 160)), ('13', (470, 160))]

subject_list = [('01', (580, 150)), ('02', (580, 150)), ('03', (580, 150))]

cwd = os.getcwd()

base_folder = r'C:\Users\USER\Perclos_k_io\Raja_Drowsy_Data_Video'

for subject_number, bb_start in subject_list:

    # Setting up directories
    base_folder = r'C:\Users\USER\PycharmProjects\Perclos_k\Current work\mp_output'

    subject_name = 'S{}'.format(subject_number)

    frame_transition = 1

    vid_number = 1
    input_vid_number = ['', r'_MD.mov', r'_MD_2.mov', r'_MD_3.mov', r'_MD_4.mov']
    input_vid_name = r'Raja_Drowsy_Data_Video/' + subject_name + r'/' + subject_name + input_vid_number[vid_number]
    output_vid_number = ['', r'_MD.mp4', r'_MD_2.mp4', r'_MD_3.mp4', r'_MD_4.mp4']
    output_vid_name = r'mp_output/' + subject_name + r'/mp_' + subject_name + output_vid_number[vid_number]

    # capturing video from input file
    cap = cv2.VideoCapture(input_vid_name)
    print(input_vid_name)

    # start of video frame (change based on last run, frame_n_end)
    # if first run set as 1
    # 3/ 15/ 28
    # 5400/ 27000/ 50400
    frame_n_start = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n_start - 1)
    frame_n_end = frame_n_start
    frame_transition_end = frame_transition
    """
    can set the desired window size here
    please leave some additional space of at least 30 pixels for height so that we can display
    frame number and other parameters

    """

    # roi_buffer = 250
    desired_window_size: Tuple[int, int] = (300, 300)  # (width, height)
    # (x, y)
    # bb_manual = [(570, 100), (570, 100), (570, 100), (570, 100)]
    # bb_start = bb_manual[vid_number]
    bb_end = (bb_start[0] + desired_window_size[0], bb_start[1] + desired_window_size[1])

    # mediapipe related variables
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=0.5)
    draw_lm = 0
    # idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
    idList = [33, 159, 133, 145, 362, 386, 263, 374]
    color = (255, 0, 255)


    p = 0
    frames_open = 0
    frames_closed = 0
    no_of_frames = 0
    minn = 0
    j = 0
    break_flag = 0
    cal_state = 1  # for calibration state (future)
    no_face = 0
    max_h = 0
    max_w = 0
    min_h = 1000
    min_w = 1000
    minute_counter = 0
    roi_color = np.zeros((300, 300))
    eyes_state = 0
    first_run = 1

    left_eye = []
    right_eye = []
    ear_list = []
    ear_avglist = []
    eyes_signal = []
    blink_signal = []
    perclos_list = [0] * 1800
    blink_dur = 0
    blink_counter = 0
    blink_update = 0
    eye_close_sum = 0

    # Set this based on how long you want to run the program
    # Default is set as 300 frames (10 seconds)
    length_of_run = 60000




    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # output_name = 'test.mp4'
    # output_name = r'mp_' + subject_name + output_vid_number[vid_number]
    out = cv2.VideoWriter(output_vid_name, fourcc, 30, (600, 600))

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels

            if break_flag == 1:
                break

            ret, image = cap.read()

            if ret:
                fps_start = time.time()

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = image[bb_start[1]:bb_end[1], bb_start[0]:bb_end[0]]
                # image = cv2.resize(image, (600, 600)) # is this necessary

                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                no_of_frames += 1

                faces = []
                if results.multi_face_landmarks:
                    for faceLms in results.multi_face_landmarks:
                        if draw_lm:
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=faceLms,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=drawing_spec,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                        face = []
                        for lm_id, lm in enumerate(faceLms.landmark):
                            ih, iw, ic = image.shape
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            face.append([x, y])
                        faces.append(face)

                        if faces:
                            face = faces[0]

                            # place landmarks
                            for id in idList:
                                cv2.circle(image, face[id], 1, color, cv2.FILLED)

                        left_eye_ratio, left_eye_left_point = get_ear([33, 159, 133, 145], face, image)
                        # Calculating right eye aspect ratio
                        right_eye_ratio, right_eye_left_point = get_ear([362, 386, 263, 374], face, image)
                        # Calculating aspect ratio for both eyes
                        ear_avg = (left_eye_ratio + right_eye_ratio) / 2
                        # ear_avg = left_eye_ratio
                        # Rounding ear_avg on two decimal places
                        # blinking_ratio_1 = ear_avg * 100
                        ear_rounded = np.round(ear_avg)
                        # blinking_ratio_rounded = ear_rounded / 100
                        # Appending blinking ratio to a list eye_blink_signal
                        ear_list.append(ear_rounded)
                        ear_avglist.append(ear_rounded)
                        if len(ear_avglist) > 3:
                            ear_avglist.pop(0)

                        # ear_plot = sum(ear_avglist) / len(ear_avglist)
                        ear_plot = statistics.mean(ear_avglist)

                        # [TODO] FIND SUITABLE THRESHOLD TO DETECT CLOSED EYES
                        if ear_avg < 20:
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

                        # PERCLOS PART HERE
                        perclos = (eye_close_sum / 1800) * 100
                        popvalue = perclos_list.pop(0)
                        perclos_list.append(eyes_state)
                        eye_close_sum = eye_close_sum + eyes_state - popvalue

                else:
                    print("no face")
                    no_face += 1

                # cv2.imshow("clone", image)
                # time.sleep(5)
                # To determine the max window size
                # if roi_color.shape[0] > max_h:
                #     max_h = roi_color.shape[0]
                #
                # if roi_color.shape[1] > max_w:
                #     max_w = roi_color.shape[1]
                #
                # if roi_color.shape[0] < min_h:
                #     min_h = roi_color.shape[0]
                #
                # if roi_color.shape[1] < min_w:
                #     min_w = roi_color.shape[1]

                # this function we feed in the cropped image and desired window size
                # output = pad_vframes(roi_color, desired_window_size)
                image = cv2.resize(image, (600, 600))
                output = image
                # image_scaled = imutils.resize(roi_color, width=220)
                text = "Frame: {}".format(frame_transition_end)
                # text coordinates are (x,y)
                cv2.putText(output, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if eyes_state == 1:
                    eye_text = 'close'
                    color = (0, 0, 255)  # red
                else:
                    eye_text = 'open'
                    color = (255, 0, 0)  # blue

                fps_end = time.time()
                fps = 1 / (fps_end - fps_start)

                text = "Eyes state: {}".format(eye_text)
                cv2.putText(output, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text = "PERCLOS: {:.2f}".format(perclos)
                cv2.putText(output, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text = "EAR: {:.2f}".format(ear_rounded)
                cv2.putText(output, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text = "BLINKS: {}".format(blink_counter)
                cv2.putText(output, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text = "FPS: {}".format(int(fps))
                cv2.putText(output, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                out.write(output)

                # cv2.imshow("image", image_scaled)

                # show the frame bounding box
                cv2.imshow("Output", output)

                frame_n_end += 1
                frame_transition_end += 1

                # this part is to stop the code when the letter 'q' is pressed on keyboard (only if cv2.imshow is used)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break_flag = 1
                    end = timer()
                    out.release()
                    cap.release()
                    print('[INFO] processing time =', timedelta(seconds=end - start))
                    break

                if frame_n_end == (frame_n_start + length_of_run):
                    break_flag = 1
                    end = timer()
                    out.release()
                    cap.release()
                    print('[INFO] processing time =', timedelta(seconds=end - start))
                    break

            else:

                end = timer()
                out.release()
                cap.release()
                print('[INFO] processing time =', timedelta(seconds=end - start))
                break

            if (no_of_frames % 1800) == 0:
                minute_counter += 1
                print('[INFO] Minutes completed =', minute_counter)
                print('[INFO] Number of frames completed =', no_of_frames)
                end_pitstop = timer()
                print('[INFO] processing time(past minute) =', timedelta(seconds=end_pitstop - start_pitstop))
                start_pitstop = end_pitstop
        # do a bit of cleanup
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print('[INFO] number of frames =', no_of_frames)
    # print('[INFO] frames start =', frame_n_start)
    # print('[INFO] frames end =', frame_n_end)
    print('[INFO] number of no_face =', no_face)
    print('[INFO] max_h =', max_h)
    print('[INFO] max_w =', max_w)
    print('[INFO] min_h =', min_h)
    print('[INFO] min_w =', min_w)
    print("DONE!")
    print(input_vid_name)
    print('[INFO - use this for next transition] frames transition end =', frame_transition_end)

print('All subjects completed')
