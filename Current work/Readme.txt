(Last Updated: 12/11/2021 8pm)
Below are the lsit of codes and explanation of their use:

1. perclos_k
Info: This will be our main code to obtain perclos values and label our dataset
Progress: Eyes open and closed state are obtained but need to tune EAR values and determine
	  PERCLOS values every minute.
Input: 	folder paths (line 30, 32, 35)
	video file name (line 237)
Tuning:	blinking ratio parameter
Optional: pickle functions (commented out)
Output: images of each frame in general folder
	images contain text (frame, eyes state)
	pickle file (optional)

2. read_no_of_frames
Info: To determine length of video (frames)
Input: 	video file name (line 3)
Output: number of frames (terminal)

3. video_crop_small
Info: To crop video dataset to smallest size, which helps to reduce processing time
Input: 	path to shape predictor file (line 24)
	video file name (line 27)
	desired_window_size (line 32) - determine after running all videos
	out(output video name) (line 36) - we standardize naming later 
	no_of_frames (line 151) - to stop the code once reach the desired frame
Output: current time (terminal) - start time
	processing time (terminal)
	number of frames (terminal)
	number of no face (terminal) - the face detection can't detect all the faces
	max_h (terminal) - max ROI height in video
	max_w (terminal) - max ROI width in video	
	min_h (terminal) - min ROI height in video
	min_w (terminal) - min ROI width in video
	cropped video file

4. read_specific_frame
Info: Basic code to just look at a particular frame
Input: 	video file name (line 3)
	frame number
Output: temporary display window of frame (press 'q' to close)

 

 	