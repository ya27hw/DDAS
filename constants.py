import os

# Hard coded index values of both eyes.
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 308, 13, 14, 17, 0]

WIDTH = 1280
HEIGHT = 720

# Adjust accordingly
EYE_RATIO_LIMIT = 0.25
EYE_RATIO_CONSECUTIVE_FRAMES = 25

MOUTH_RATIO_LIMIT = 1.5
MOUTH_RATIO_CONSECUTIVE_FRAMES = 25

# Load file
SHAPE_PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), 'resources/shape_predictor_68_face_landmarks.dat')

# Change if required
CAM_INDEX = 0