import cv2
import numpy as np
from utils import eye_aspect_ratio, mouth_aspect_ratio, extract_points, draw_hull
import constants
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh

# def euclidean(a, b):
#     return np.linalg.norm(a - b)


# Used to keep track of how many frames the eye/mouth has exceeded limits
eye_counter = 0
mouth_counter = 0
eye_alarm = False
mouth_alarm = False

# Starting the live video stream
video_stream = cv2.VideoCapture(constants.CAM_INDEX)
# Set the width (CAP_PROP_FRAME_WIDTH = 3)
video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, constants.WIDTH)

# Set the height (CAP_PROP_FRAME_HEIGHT = 4)
video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, constants.HEIGHT)

if not video_stream.isOpened():
    raise Exception(
        "Could not open video stream. Please check your camera index")
    exit(1)

print("Starting camera... Press 'ESC' to quit.")

# Main loop for the application
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
) as face_mesh:
    while True:
        # Read the next frame from the stream
        ret, frame = video_stream.read()
        if not ret:
            print("Error reading frame.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        if not results.multi_face_landmarks:
            continue

        lm = results.multi_face_landmarks[0].landmark

        left_eye_points = extract_points(lm, constants.LEFT_EYE_IDX)
        right_eye_points = extract_points(lm, constants.RIGHT_EYE_IDX)
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        ear = (left_ear + right_ear) / 2
        mar = mouth_aspect_ratio(lm)

        if ear < constants.EYE_RATIO_LIMIT:
            eye_counter += 1
            if eye_counter >= constants.EYE_RATIO_CONSECUTIVE_FRAMES:
                if not eye_alarm:
                    eye_alarm = True
                print("DROWSINESS ALERT!")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_counter = 0
            eye_alarm = False

        if mar > constants.MOUTH_RATIO_LIMIT:
            mouth_counter += 1
            if mouth_counter >= constants.MOUTH_RATIO_CONSECUTIVE_FRAMES:
                if not mouth_alarm:
                    mouth_alarm = True
                print("DROWSINESS ALERT!")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            mouth_counter = 0
            mouth_alarm = False

        # Draw eyes
        draw_hull(cv2, frame, lm, constants.LEFT_EYE_IDX)
        draw_hull(cv2, frame, lm, constants.RIGHT_EYE_IDX)
        draw_hull(cv2, frame, lm, constants.MOUTH_IDX)

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("DDAS - Face Tracking",
                   cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) == 27:
            break


# Outside the main loop, this code runs when the application is closed.
cv2.destroyAllWindows()
video_stream.release()
print("Done.")
