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