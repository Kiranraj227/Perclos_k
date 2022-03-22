import os
import cv2

import shutil
from typing import Tuple


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


length_of_run = 300

subject_list = [('01', (580, 150)), ('02', (580, 150)), ('03', (580, 150))]

cwd = os.getcwd()

for subject_number, bb_start in subject_list:

    # Initialise variables for each subject
    frame_counter = 0

    desired_window_size: Tuple[int, int] = (300, 300)  # (width, height)
    # (x, y)
    # bb_manual = [(570, 100), (570, 100), (570, 100), (570, 100)]
    # bb_start = bb_manual[vid_number]
    bb_end = (bb_start[0] + desired_window_size[0], bb_start[1] + desired_window_size[1])

    # Setting up directories
    base_folder = r'C://Users//USER//Perclos_k_io'

    subject_name = 'S{}'.format(subject_number)

    frame_transition = 1

    input_vid_number = [r'_EC.mov', r'_MD.mov', r'_MD_2.mov', r'_MD_3.mov', r'_MD_4.mov', r'_EO.mov']
    # input_vid_name = r'Raja_Drowsy_Data_Video/' + subject_name + r'/' + subject_name + input_vid_number[vid_number]
    output_vid_number = [r'_EC.mp4', r'_MD.mp4', r'_MD_2.mp4', r'_MD_3.mp4', r'_MD_4.mp4', r'_EO.mp4']
    # output_vid_name = r'mp_output/' + subject_name + r'/mp_' + subject_name + output_vid_number[vid_number]

    for vid_number in range(6):

        break_flag = 0

        input_vid_name = r'Raja_Drowsy_Data_Video/' + subject_name + r'/' + subject_name + input_vid_number[vid_number]
        input_vid_path = os.path.join(base_folder, input_vid_name)

        if not os.path.exists(input_vid_path):
            continue

        output_vid_dir = r'try_output//' + subject_name
        output_vid_name = output_vid_dir + r'//mp_' + subject_name + output_vid_number[vid_number]
        output_vid_path = os.path.join(base_folder, output_vid_dir)

        # checks whether the output folder exists
        check_make_folder(output_vid_path, output_vid_dir)

        # full path that also contains name of output video
        full_output_vid_path = os.path.join(base_folder, output_vid_name)

        os.chdir(output_vid_path)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_vid_name = "mp_" + subject_name + output_vid_number[vid_number]
        out = cv2.VideoWriter("{}".format(out_vid_name), fourcc, 30, (600, 600))

        cap = cv2.VideoCapture(input_vid_path)
        print(input_vid_name)

        while True:

            if break_flag == 1:
                break

            ret, frame = cap.read()

            if ret:

                frame_counter += 1
                crop_roi = frame[bb_start[1]:bb_end[1], bb_start[0]:bb_end[0]]

                output = cv2.resize(crop_roi, (600, 600))

                text = "Frame: {}".format(frame_counter)
                # text coordinates are (x,y)
                cv2.putText(output, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("Output", output)

                out.write(output)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    print('user ended it at frame {}, next'.format(frame_counter))
                    out.release()
                    cap.release()
                    break_flag = 1
                    break

                if frame_counter == length_of_run:
                    print('{} frames done, next'.format(length_of_run))
                    out.release()
                    cap.release()
                    break_flag = 1
                    break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

print('all done')
