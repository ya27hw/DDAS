import numpy as np
from scipy.spatial import distance as dist


def draw_hull(cv2, frame, landmarks, idx, color=(0, 255, 0)):
    """
    Draws a convex hull around a set of points on a frame.

    Parameters
    ----------
    cv2 : object
        The OpenCV library object.
    frame : numpy.ndarray
        The frame on which to draw the convex hull.
    landmarks : dlib.full_object_detection
        The face landmarks object from which to extract points.
    idx : list
        The indices of the points to extract.
    color : tuple, optional
        The color of the convex hull. Defaults to (0, 255, 0), i.e. green.

    Returns
    -------
    None
    """

    height, width = frame.shape[:2]
    points = np.array(
        [[int(landmarks[i].x * width), int(landmarks[i].y * height)] for i in idx])
    hull = cv2.convexHull(points)
    cv2.drawContours(frame, [hull], -1, color, 1)


def extract_points(face_landmarks, idx) -> np.ndarray:
    """
    Extract points from a face landmarks object given an index.

    Parameters
    ----------
    face_landmarks : dlib.full_object_detection
        The face landmarks object from which to extract points
    idx : list
        The indices of the points to extract

    Returns
    -------
    numpy.ndarray
        A numpy array containing the extracted points
    """
    return np.array([[face_landmarks[i].x, face_landmarks[i].y] for i in idx])


def eye_aspect_ratio(points) -> float:
    """
    Calculate the aspect ratio of the eye region.

    The aspect ratio of the eye region is calculated as the ratio
    of the sum of the distances between the points of the eye
    region to the distance between the outer points of the eye
    region.

    Parameters
    ----------
    points : array-like
        The landmark representing the eye region

    Returns
    -------
    float
        The aspect ratio of the eye region
    """
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth) -> float:
    """
    Calculate the aspect ratio of the mouth region.

    The aspect ratio of the mouth region is calculated as the ratio
    of the distance between the upper and lower lip points to the
    distance between the left and right lip points.

    Parameters
    ----------
    mouth : array-like
        The landmark representing the mouth region

    Returns
    -------
    float
        The aspect ratio of the mouth region
    """
    upper = np.array([mouth[13].x, mouth[13].y])
    lower = np.array([mouth[14].x, mouth[14].y])
    right = np.array([mouth[308].x, mouth[308].y])
    left = np.array([mouth[78].x, mouth[78].y])
    return dist.euclidean(upper, lower) / dist.euclidean(left, right)
