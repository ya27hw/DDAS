from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    # C = dist.euclidean(mouth[3], mouth[5])
    C = dist.euclidean(mouth[0], mouth[6])
    return ( A + B) / (2.0 * C)