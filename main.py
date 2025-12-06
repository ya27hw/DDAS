import cv2
import dlib
from imutils import face_utils, resize
from utils import eye_aspect_ratio, mouth_aspect_ratio
import constants

# Init Dlib components
# First, the HOG detector, whcih finds the bounding boxes of the face
detector = dlib.get_frontal_face_detector()

# Second, the shape predictor. This maps the 68 xy coords onto the face
predictor = dlib.shape_predictor(constants.SHAPE_PREDICTOR_PATH)

# Used to keep track of how many frames the eye/mouth has exceeded limits
eye_counter = 0
mouth_counter = 0
eye_alarm = False
mouth_alarm = False

# Get the landmarks of the left and right eye
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Starting the live video stream
video_stream = cv2.VideoCapture(constants.CAM_INDEX)
# Set the width (CAP_PROP_FRAME_WIDTH = 3)
video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, constants.WIDTH)

# Set the height (CAP_PROP_FRAME_HEIGHT = 4)
video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, constants.HEIGHT)

if not video_stream.isOpened():
    raise Exception("Could not open video stream. Please check your camera index")
    exit(1)

print("Starting camera... Press 'q' to quit.")

# Main loop for the application
while True:
    # Read the next frame from the stream
    ret, frame = video_stream.read()
    if not ret:
        print("Error reading frame.")
        break

    frame = resize(frame, width=constants.WIDTH)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = detector(gray_frame, 1)

    for face in faces:
        # Get the face landmarks from the shape predictor
        face_landmarks = predictor(gray_frame, face)
        # Convert to numpy array to read easier
        face_landmarks = face_utils.shape_to_np(face_landmarks)

        left_eye = face_landmarks[leftEyeStart:leftEyeEnd]
        right_eye = face_landmarks[rightEyeStart:rightEyeEnd]
        mouth = face_landmarks[mouthStart:mouthEnd]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2       
        mar = mouth_aspect_ratio(mouth)

        if ear < constants.EYE_RATIO_LIMIT:
            eye_counter +=1
            if eye_counter >= constants.EYE_RATIO_CONSECUTIVE_FRAMES:
                if not eye_alarm:
                    eye_alarm = True
                print("DROWSINESS ALERT!")
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            eye_counter = 0
            eye_alarm = False

        if mar > constants.MOUTH_RATIO_LIMIT:
            mouth_counter +=1
            if mouth_counter >= constants.MOUTH_RATIO_CONSECUTIVE_FRAMES:
                if not mouth_alarm:
                    mouth_alarm = True
                print("YAWNING ALERT!")
                cv2.putText(frame, "YAWNING ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            mouth_counter = 0
            mouth_alarm = False

        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        mouth_hull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

       
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw the face landmarks
        for (x, y) in face_landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        cv2.imshow("DDAS System", frame)

        # Listen to key presses from 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break



# Outside the main loop, this code runs when the application is closed.
cv2.destroyAllWindows()
video_stream.release()
print("Done.")