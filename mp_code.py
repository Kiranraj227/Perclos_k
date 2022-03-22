import cv2
import mediapipe as mp
import time

desired_window_size = (300, 300)  # (width, height)
# (x, y)
# bb_start = (570, 50) # for S01
bb_start = (570, 100)  # for S03
bb_end = (bb_start[0] + desired_window_size[0], bb_start[1] + desired_window_size[1])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=0.5, circle_radius=0.5)

# vid_name = 'test_video.mp4'
vid_name = 'S03_MD.mov'
cap = cv2.VideoCapture(vid_name)
# For webcam input:
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while True:
        success, image = cap.read()

        start = time.time()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image[bb_start[1]:bb_end[1], bb_start[0]:bb_end[0]]
        image = cv2.resize(image, (600, 600))

        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #         .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('MediaPipe FaceMesh', image)
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("you ended it")
            # break_flag = 1
            break

cap.release()
cv2.destroyAllWindows()
