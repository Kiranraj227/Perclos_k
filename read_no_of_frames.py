import cv2

cap = cv2.VideoCapture('cropped.mp4')
# initialize the total number of frames read
total = 0
# loop over the frames of the video
while True:
    # grab the current frame
    (grabbed, frame) = cap.read()

    # check to see if we have reached the end of the
    # video
    if not grabbed:
        break
    # increment the total number of frames read
    total += 1
    print(total)
