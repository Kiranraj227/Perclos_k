import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot



break_flag = 0
EARList = []
blinkCounter = 0
counter = 0


desired_window_size = (300, 300)  # (width, height)
# (x, y)
bb_start = (470, 160)
# bb_start = (470, 200)
bb_end = (bb_start[0] + desired_window_size[0], bb_start[1] + desired_window_size[1])


# vid_name = 'test_video.mp4'
vid_name = r'Raja_Drowsy_Data_Video/S13/S13_MD.mov'
# vid_name = 'S03_MD.mp4'
cap = cv2.VideoCapture(vid_name)
detector = FaceMeshDetector(maxFaces=1)


plotY = LivePlot(600, 600, [10, 50], invert=True)
# idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
idList = [33, 159, 133, 145, 362, 386, 263, 374]
color = (255, 0, 255)
while True:

    ret, img = cap.read()
    # img = img[bb_start[0]:bb_start[0] + bb_size[0], bb_start[1]:bb_start[1] + bb_size[1]]

    if ret:
        img = img[bb_start[1]:bb_end[1], bb_start[0]:bb_end[0]]
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            num = 0
            for id in idList:
                # num +=1
                # cv2.putText(img, num, (face[id][0], face[id][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(img, face[id], 1, color, cv2.FILLED)

            leftUp = face[159]
            leftDown = face[145]
            leftLeft = face[33]
            leftRight = face[133]
            lengthVer, _ = detector.findDistance(leftUp, leftDown)
            lengthHor, _ = detector.findDistance(leftLeft, leftRight)
            cv2.line(img, leftUp, leftDown, (0, 200, 0), 1)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 1)

            EAR = (lengthVer/lengthHor)*100
            EARList.append(EAR)
            if len(EARList) > 3:
                EARList.pop(0)
            EAR_Avg = sum(EARList) / len(EARList)

            # BLINK COUNTER
            if EAR_Avg < 20 and counter == 0:
                blinkCounter += 1
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (10, 20), scale=1, thickness=2,
                               colorT=(255, 255, 255), colorR=(255, 0, 255))
            imgPlot = plotY.update(EAR_Avg)
            cv2.imshow("ImagePlot", imgPlot)

        img = cv2.resize(img, (600, 600))

        cv2.imshow("image", img)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("you ended it")
            break_flag = 1
            break
    elif break_flag == 1:
        break

    else:
        print('end of video')
        break

cap.release()
cv2.destroyAllWindows()
