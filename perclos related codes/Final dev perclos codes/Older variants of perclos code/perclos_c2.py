'''
This revision is based on
http://datahacker.rs/011-how-to-detect-eye-blinking-in-videos-using-dlib-and-opencv-in-python/
'''

# import the necessary packages

from imutils import face_utils
import numpy as np
import dlib
import cv2
import os
import shutil
import pickle

import time
from datetime import datetime

'''
https://www.youtube.com/watch?v=RvfF9CDzn1s
'''
# Total of 6291 pages
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

save_pickle_name = 'oatd_complete_all_specific_page'
dir_open = '../eyes_open'
dir_close = '../eyes_close'
general_folder = 'general_folder'
# base_folder = r'C:\Users\rpb\Desktop\drowsiness_image_storage'
base_folder = r'C:\Users\USER\PycharmProjects\Perclos'
# src_out = 'THIS PATH IS LOCATED OUTSIDE OR SOMEWHERE ELSE, IF UNAVAILABLE, JUST DOWNLOAD'
# dlib_path = src_out + 'shape_predictor_68_face_landmarks.dat'
dlib_path = '../../Current work/shape_predictor_68_face_landmarks.dat'

x = 1


def create_dir(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):

        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        shutil.rmtree(dirName, ignore_errors=False)
        os.mkdir(dirName)

        print("Directory ", dirName, " already exists")


full_dir_path_open = os.path.join(base_folder, dir_open)
full_dir_path_close = os.path.join(base_folder, dir_close)
full_dir_path_general_folder = os.path.join(base_folder, general_folder)

# create_dir(full_dir_path_open)
# create_dir(full_dir_path_close)


# Initializing the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

'''
From the bold guy tutorial
vvvvv
'''
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


class eye_close_state():
    def __init__(self, path):
        self.path = path
        self.assign_eye_close_open()

    # Defining the mid-point
    def midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    # Defining the Euclidean distance
    def euclidean_distance(self, leftx, lefty, rightx, righty):
        return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)

    # Defining the eye aspect ratio
    def get_ear(self, eye_points, facial_landmarks, frame):
        # Defining the left point of the eye
        left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
        # Defining the right point of the eye
        right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
        # Defining the top mid-point of the eye
        center_top = self.midpoint(facial_landmarks.part(eye_points[1]),
                                   facial_landmarks.part(eye_points[2]))
        # Defining the bottom mid-point of the eye
        center_bottom = self.midpoint(facial_landmarks.part(eye_points[5]),
                                      facial_landmarks.part(eye_points[4]))

        # Drawing horizontal and vertical line
        hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0),
                            3)
        ver_line = cv2.line(frame, (center_top[0], center_top[1]), (center_bottom[0], center_bottom[1]),
                            (255, 0, 0), 3)
        # Calculating length of the horizontal and vertical line
        hor_line_lenght = self.euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
        ver_line_lenght = self.euclidean_distance(center_top[0], center_top[1], center_bottom[0],
                                                  center_bottom[1])
        # Calculating eye aspect ratio
        ear = ver_line_lenght / hor_line_lenght

        return ear, left_point

    def save_chunck_file(self, try_append_x, save_pickle_name):
        save_pickle_name_folder = os.path.join(base_folder, general_folder, save_pickle_name)
        # with open(f'{oatd_search_result_fname}.txt', 'r') as file:
        with open(f'{save_pickle_name_folder}.pickle', 'wb') as handle:
            pickle.dump(try_append_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Function to extract frames
    def assign_eye_close_open(self):
        path = self.path
        start_time = time.time()
        # Path to video file
        vid_obj = cv2.VideoCapture(path)

        last_page = int(vid_obj.get(cv2.CAP_PROP_FRAME_COUNT))
        x = 1
        perct_prog_previou = []
        # Creating a list eye_blink_signal
        eye_blink_signal = []
        # Creating an object blink_ counter
        blink_counter = 0
        previous_ratio = 100
        # Creating a while loop
        count = 0
        counter_file_name = 0
        try_append = []
        initial_decision = []
        while True:
            ret, frame = vid_obj.read()
            if not ret:
                break
            # Converting a color frame into a grayscale frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Creating an object in which we will sore detected faces
            faces = detector(gray)
            for face in faces:
                c_left_x, c_top_y = face.left(), face.top()
                c_right_x1, c_bottom_y1 = face.right(), face.bottom()
                coordinate_face = dict(c_left_x=c_left_x, c_top_y=c_top_y, c_right_x1=c_right_x1,
                                       c_bottom_y1=c_bottom_y1)
                # Creating an object in which we will sore detected facial landmarks
                landmarks = predictor(gray, face)
                # Calculating left eye aspect ratio
                left_eye_ratio, left_eye_left_point = eye_close_state.get_ear(self, [36, 37, 38, 39, 40, 41],
                                                                              landmarks, frame)
                # Calculating right eye aspect ratio
                right_eye_ratio, right_eye_left_point = eye_close_state.get_ear(self, [42, 43, 44, 45, 46, 47],
                                                                                landmarks, frame)
                # Calculating aspect ratio for both eyes
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                # Rounding blinking_ratio on two decimal places
                blinking_ratio_1 = blinking_ratio * 100
                blinking_ratio_2 = np.round(blinking_ratio_1)
                blinking_ratio_rounded = blinking_ratio_2 / 100
                # Appending blinking ratio to a list eye_blink_signal
                eye_blink_signal.append(blinking_ratio)


                if blinking_ratio < 0.20:
                    state = 'Close'
                else:
                    '''
                    Otherwise, Line 116 handles the case where the eye aspect ratio is not below the blink threshold.
                    In this case, we make another check on Line 119 to see if a sufficient number of consecutive frames 
                    contained an eye blink ratio below our pre-defined threshold.

                    If the check passes, we increment the TOTAL number of blinks (Line 120).
                    We then reset the number of consecutive blinks COUNTER (Line 123).

                    '''
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    state = 'Open'

                '''
                From the bold guy tutorial
                vvvvv
                '''
                cv2.putText(gray, "Frame: {:.2f}".format(counter_file_name), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(gray, "Auto Sort: {}".format(state), (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                data = dict(cv_image=gray, frame_NO=2, c=3)
                try_append.append(gray)
                cv2.imwrite(os.path.join(full_dir_path_general_folder, "frame%d.jpg" % counter_file_name), gray)
                init_result = dict(cv_image=gray, state=state, frame_no=counter_file_name,
                                   coordinate_face=coordinate_face, left_eye_left_point=left_eye_left_point)
                initial_decision.append(init_result)

                perct_prog = int(round((counter_file_name / last_page) * 100, 1))
                if perct_prog_previou != perct_prog or perct_prog == 100:
                    perct_prog_previou = perct_prog

                    if counter_file_name == 0:
                        print(" 0 percent complete")

                    if perct_prog % 5 == 0 and perct_prog != 0:
                        hours, rem = divmod(time.time() - start_time, 3600)
                        elapsed_time_minutes, seconds = divmod(rem, 60)
                        if int(elapsed_time_minutes) == 0.00:
                            print(f"{perct_prog:d} percents complete and elapsed {seconds:.2f} seconds")
                        else:
                            now = datetime.now()
                            current_time = now.strftime("%H:%M:%S")
                            # print("Current Time =", current_time)
                            print(
                                f"{perct_prog:d} percents complete and elapsed {elapsed_time_minutes:.2f} minute and current time is {current_time}")
                            perct_prog_str = str(perct_prog)
                            self.save_chunck_file(initial_decision, perct_prog_str)
                            initial_decision = []
                            break

                counter_file_name = counter_file_name + 1

        my_last_line = counter_file_name
        print(counter_file_name)

        vid_obj.release()


# eye_close_state(r'my_video.mov')
eye_close_state(r'0.mp4')