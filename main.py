import os
import cv2
import dlib
from imutils import face_utils

# Load file
SHAPE_PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'resources/shape_predictor_68_face_landmarks.dat')

# Change if required
CAM_INDEX = 0

# Init Dlib components
# First, the HOG detector, whcih finds the bounding boxes of the face
detector = dlib.get_frontal_face_detector()

# Second, the shape predictor. This maps the 68 xy coords onto the face
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Starting the live video stream
# This will refresh every 1ms
video_stream = cv2.VideoCapture(CAM_INDEX)
if not video_stream.isOpened():
    raise Exception("Could not open video stream. Please check your camera index", CAM_INDEX)
    exit(1)

print("Starting camera... Press 'q' to quit.")

# Main loop for the application
while True:
    # Read the next frame from the stream
    ret, frame = video_stream.read()
    if not ret:
        print("Error reading frame.")
        break

    # Adjust frame size to 450x250 for faster operation, and use grayscale as well
    frame = cv2.resize(frame, (450, 250))
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = detector(gray_frame, 0)

    for face in faces:
        # Get the face landmarks from the shape predictor
        face_landmarks = predictor(gray_frame, face)

        # Then convert it to a numpy array for reading
        face_landmarks_np = face_utils.shape_to_np(face_landmarks)

        # Draw the face landmarks
        for (x, y) in face_landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        cv2.imshow("Title", frame)

        # Listen to key presses from 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Outside the main loop, this code runs when the application is closed.
cv2.destroyAllWindows()
video_stream.release()
print("Done.")
