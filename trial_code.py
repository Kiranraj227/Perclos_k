import os

subject_list = [('03', (580, 150)), ('05', (1, 1)), ('08', (2, 2)), ('11', (3, 3)), ('13', (4, 4))]

for subject_number, bb_start in subject_list:

    # Setting up directories
    base_folder = r'C:/Users/USER/PycharmProjects/Perclos_k/Current work/mp_output'

    subject_name = 'S{}'.format(subject_number)

    frame_transition = 1

    vid_number = 1
    input_vid_number = ['', r'_MD.mov', r'_MD_2.mov', r'_MD_3.mov', r'_MD_4.mov']
    input_vid_name = r'Raja_Drowsy_Data_Video/' + subject_name + r'/' + subject_name + input_vid_number[vid_number]
    output_vid_number = ['', r'_MD.mp4', r'_MD_2.mp4', r'_MD_3.mp4', r'_MD_4.mp4']
    output_vid_name = r'/' + r'mp_output/' + r'mp_' + subject_name + output_vid_number[vid_number]
    # full_dir_output = os.path.join(base_folder, output_vid_name)
    # vid_name = 'test_video.mp4'
    # vid_name = 'S03_MD.mov'
    # capturing video from input file

    print(input_vid_name)
    print(output_vid_name)
    print('           ')
print('done!')
