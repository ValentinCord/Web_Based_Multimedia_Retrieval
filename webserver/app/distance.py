import numpy as np
import cv2
import math

def distance_matching(cfg, distance_vector, distance_matrix, descriptors):
    # matching vector distance name with function
    if distance_vector == 'euclidean':
        distance_vector_func = euclidean
    elif distance_vector == 'chiSquareDistance':
        distance_vector_func = chiSquareDistance
    elif distance_vector == 'bhatta':
        distance_vector_func = bhatta
    elif distance_vector == 'correlation':
        distance_vector_func = correlation
    elif distance_vector == 'interesection':
        distance_vector_func = interesection

    # matching matrix distance name with function
    if distance_matrix == 'flann':
        distance_matrix_func = flann
    elif distance_matrix == 'bruteForceMatching':
        distance_matrix_func = bruteForceMatching

    # matching of descriptors with right distance
    distance_func = dict()
    for desc_name in descriptors:
        distance_func[desc_name] = distance_vector_func if desc_name in cfg['vector'] else distance_matrix_func

    return distance_func

def euclidean(l1, l2):
    n = min(len(l1), len(l2))
    return np.sqrt(np.sum((l1[:n] - l2[:n])**2))

def chiSquareDistance(l1, l2):
    n = min(len(l1), len(l2))
    return np.sum((l1[:n] - l2[:n])**2 / l2[:n])

def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1,l2,dtype=np.float64)),dtype=np.float64)
    den = np.sqrt(np.sum(l1,dtype=np.float64)*np.sum(l2,dtype=np.float64))
    return math.sqrt( 1 - num / den )

def flann(a,b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    if a.shape[0]==0 or b.shape[0]==0:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)

def correlation(a, b):
    return cv2.compareHist(np.float32(a), np.float32(b), cv2.HISTCMP_CORREL)

def interesection(a, b):
    return cv2.compareHist(np.float32(a), np.float32(b), cv2.HISTCMP_INTERSECT)
