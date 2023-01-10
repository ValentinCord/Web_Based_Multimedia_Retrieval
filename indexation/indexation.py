import os
import cv2
import numpy as np
import json

from time import time
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, local_binary_pattern, graycoprops

def main():

    time_SIFT = generate(generateSIFT, 'webserver/app/static/db', 'SIFT.json')
    time_HSV  = generate(generateHistogramme_HSV, 'webserver/app/static/db', 'HSV.json')
    time_HOG  = generate(generateHOG, 'webserver/app/static/db', 'HOG.json')
    time_LBP  = generate(generateLBP, 'webserver/app/static/db', 'LBP.json')
    time_ORB  = generate(generateORB, 'webserver/app/static/db', 'ORB.json')
    time_BGR  = generate(generateHistogramme_Color, 'webserver/app/static/db', 'BGR.json')
    time_GLCM = generate(generateGLCM, 'webserver/app/static/db', 'GLCM.json')

    with open('benchmark.txt', 'w') as fin:
        fin.write('Computation times for descriptors benchmark\n')
        fin.write(f'BGR  : {time_BGR} s\n')
        fin.write(f'HSV  : {time_HSV} s\n')
        fin.write(f'HOG  : {time_HOG} s\n')
        fin.write(f'LBP  : {time_LBP} s\n')
        fin.write(f'ORB  : {time_ORB} s\n')
        fin.write(f'SIFT : {time_SIFT} s\n')
        fin.write(f'GLCM : {time_GLCM} s\n')

def generate(generate_func, source, output):
    start = time()
    feature = generate_func(source)
    save(feature, output)
    del feature
    print(f'[INFO] Indexation : {generate_func.__name__} --> Done')
    return round(time() - start, 3)

def save(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, cls = JsonCustomEncoder)

def generateHistogramme_Color(folder):
    data = list()
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        histB = cv2.calcHist([img], [0], None, [256], [0,256])
        histG = cv2.calcHist([img], [1], None, [256], [0,256])
        histR = cv2.calcHist([img], [2], None, [256], [0,256])
        feature = np.concatenate((histB, np.concatenate((histG, histR), axis = None)), axis = None)
        num_image, _ = path.split('.')
        data.append({num_image : feature})
    return data

def generateHistogramme_HSV(folder):
    data = list()
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img], [0], None, [256], [0,256])
        histS = cv2.calcHist([img], [1], None, [256], [0,256])
        histV = cv2.calcHist([img], [2], None, [256], [0,256])
        feature = np.concatenate((histH, np.concatenate((histS,histV), axis = None)), axis = None)
        num_image, _ = path.split(".")
        data.append({num_image : feature})
    return data
   
def generateSIFT(folder):
    data = list()
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        w, h, c = img.shape
        new_size = (int(w*0.3), int(h*0.3))
        img = cv2.resize(img, new_size)
        sift = cv2.SIFT_create()  
        kps, descriptors = sift.detectAndCompute(img, None)
        num_image, _ = path.split(".")
        if descriptors is not None:
            data.append({num_image : descriptors})
    return data

def generateORB(folder):
    data = list()
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        orb = cv2.ORB_create()
        kps, descriptors = orb.detectAndCompute(img, None)
        num_image, _ = path.split(".")
        if descriptors is not None:        
            data.append({num_image : descriptors})
    return data

def generateGLCM(folder):
    distances = [1, -1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    data = list()
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img_as_ubyte(gray)
        glcmMatrix = graycomatrix(gray, distances = distances, angles = angles, normed = True)
        glcmProperties1 = graycoprops(glcmMatrix, 'contrast').ravel()
        glcmProperties2 = graycoprops(glcmMatrix, 'dissimilarity').ravel()
        glcmProperties3 = graycoprops(glcmMatrix, 'homogeneity').ravel()
        glcmProperties4 = graycoprops(glcmMatrix, 'energy').ravel()
        glcmProperties5 = graycoprops(glcmMatrix, 'correlation').ravel()
        glcmProperties6 = graycoprops(glcmMatrix, 'ASM').ravel()
        feature =np.array([glcmProperties1,glcmProperties2,glcmProperties3,glcmProperties4,glcmProperties5,glcmProperties6]).ravel()
        num_image, _ = path.split(".")
        data.append({num_image : feature})
    return data
        
def generateLBP(folder):
    points = 8
    radius = 1
    method = 'default'
    subSize = (70,70)
    data = list()
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (350,350))
        fullLBPmatrix = local_binary_pattern(img, points, radius, method)
        histograms = []
        for i in range(int(fullLBPmatrix.shape[0]/subSize[0])):
            for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                subVector = fullLBPmatrix[i*subSize[0]:(i+1)*subSize[0], j*subSize[1]:(j+1)*subSize[1]].ravel()
                subHist, edges = np.histogram(subVector, bins = int(2**points), range = (0,2**points))
                histograms = np.concatenate((histograms, subHist), axis = None)
        num_image, _ = path.split(".")
        data.append({num_image : histograms})
    return data

def generateHOG(folder):
    data = list()
    cellSize = (25,25)
    blockSize = (50,50)
    blockStride = (25,25)
    nBins = 9
    winSize = (350,350)
    for path in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, path))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, winSize)
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins)
        feature = hog.compute(image)
        num_image, _ = path.split(".")
        data.append({num_image : feature})
    return data

class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()