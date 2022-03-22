import cv2


print("[INFO] starting video stream...")

# subject_list = [('03', (580, 150)), ('05', (580, 150)), ('08', (470, 200)), ('11', (470, 160)), ('13', (470, 160))]

# for subject_number, bb_start in subject_list:

subject_number = '07'
bb_start = (600, 180)

# Setting up directories
base_folder = r'C:\Users\USER\PycharmProjects\Perclos_k\Current work\mp_output'

subject_name = 'S{}'.format(subject_number)

frame_transition = 1

vid_number = 1
input_vid_number = ['', r'_MD.mov', r'_MD_2.mov', r'_MD_3.mov', r'_MD_4.mov']
input_vid_name = r'Raja_Drowsy_Data_Video/' + subject_name + r'/' + subject_name + input_vid_number[vid_number]
output_vid_number = ['', r'_MD.mp4', r'_MD_2.mp4', r'_MD_3.mp4', r'_MD_4.mp4']
output_vid_name = r'mp_output/' + subject_name + r'/mp_' + subject_name + output_vid_number[vid_number]
# full_dir_output = os.path.join(base_folder, output_vid_name)
# vid_name = 'test_video.mp4'
# vid_name = 'S03_MD.mov'
# capturing video from input file
cap = cv2.VideoCapture(input_vid_name)
print(input_vid_name)



# cap = cv2.VideoCapture('S03_MD.mov')
# cap = cv2.VideoCapture('cropped.mp4')

# (y,x)
bb_size = (300, 300)
# bb_start = (50, 570)
# bb_start = (100, 570)
# loop over the frames of the video
while True:
    frame_n = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n-1)
    # grab the current frame
    (grabbed, frame) = cap.read()
    if grabbed:
        # cv2.imshow('video', frame)
        bb = frame[bb_start[1]:bb_start[1]+bb_size[1], bb_start[0]:bb_start[0]+bb_size[0]]
        cv2.imshow('CROP', bb)
        print("grabbed")
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        print("not grabbed")
        break
cap.release()
cv2.destroyAllWindows()
print("done")
# print("all done")
