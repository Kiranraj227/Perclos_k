'''
This revision is based on
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
'''

# import the necessary packages
import numpy as np
# import imutils
# import time
import cv2

video = "blink_video.mp4"
cap = cv2.VideoCapture(video)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# output_name = 'test.mp4'
out = cv2.VideoWriter('output.mp4', fourcc, 30, (frame_width, frame_height))

# loop over frames from the video stream
while True:
    ret, frame = cap.read()
    if ret:

        output = np.zeros((frame_height, frame_width, 3), dtype="uint8")

        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(fps_cap, w, h)

        # output = np.zeros((frame_height * 2, frame_width * 2, 3), dtype="uint8")
        # output[0:frame_height, 0:frame_width] = frame
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output[0:frame_height, 0:frame_width] = frame
        # write the output frame to file
        out.write(output)

        # show the frames
        cv2.imshow("Output", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# do a bit of cleanup
print("[INFO] cleaning up...")
cap.release()
out.release()
cv2.destroyAllWindows()
