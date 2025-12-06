import os

WIDTH = 1280
HEIGHT = 720

# Adjust accordingly
EYE_RATIO_LIMIT = 0.25
EYE_RATIO_CONSECUTIVE_FRAMES = 10

MOUTH_RATIO_LIMIT = 0.55
MOUTH_RATIO_CONSECUTIVE_FRAMES = 10

# Load file
SHAPE_PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'resources/shape_predictor_68_face_landmarks.dat')

# Change if required
CAM_INDEX = 0