import cv2

cap = cv2.VideoCapture('S06_20170817_032034.mov')
# cap = cv2.VideoCapture('cropped.mp4')

# loop over the frames of the video
while True:
    frame_n = 269
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n-1)
    # grab the current frame
    (grabbed, frame) = cap.read()
    if grabbed:
        cv2.imshow('video', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
print("done")
