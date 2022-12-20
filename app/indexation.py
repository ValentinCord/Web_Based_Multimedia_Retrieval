import os
import cv2
import numpy as np

from time import time
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def main():

    s1 = time()
    generateHistogramme_Color("db/araignees")
    generateHistogramme_Color("db/chiens")
    generateHistogramme_Color("db/oiseaux")
    e1 = time()
    
    s2 = time()
    generateHistogramme_HSV("db/araignees")
    generateHistogramme_HSV("db/chiens")
    generateHistogramme_HSV("db/oiseaux")
    e2 = time()

    s3 = time()
    generateHOG("db/araignees")
    generateHOG("db/chiens")
    generateHOG("db/oiseaux")
    e3 = time()

    s4 = time()
    generateLBP("db/araignees")
    generateLBP("db/chiens")
    generateLBP("db/oiseaux")
    e4 = time()

    s5 = time()
    generateORB("db/araignees")
    generateORB("db/chiens")
    generateORB("db/oiseaux")
    e5 = time()

    s6 = time()
    generateSIFT("db/araignees")
    generateSIFT("db/chiens")
    generateSIFT("db/oiseaux")
    e6 = time()

    s7 = time()
    generateGLCM("db/araignees")
    generateGLCM("db/chiens")
    generateGLCM("db/oiseaux")
    e7 = time()

    bgr  = round(e1-s1, 3)
    hsv  = round(e2-s2, 3)
    hog  = round(e3-s3, 3)
    lbp  = round(e4-s4, 3)
    orb  = round(e5-s5, 3)
    sift = round(e6-s6, 3)
    glcm = round(e7-s7, 3)

    with open("benchmark.txt", "w") as fin:
        fin.write('Temps pour calculer les descripteurs :' + '\n')
        fin.write(f'BGR  : {bgr} s\n')
        fin.write(f'HSV  : {hsv} s\n')
        fin.write(f'HOG  : {hog} s\n')
        fin.write(f'LBP  : {lbp} s\n')
        fin.write(f'ORB  : {orb} s\n')
        fin.write(f'SIFT : {sift} s\n')
        fin.write(f'GLCM : {glcm} s\n')


def status_log(function, folder):
    print("[INFO] Indexation : " + function.__name__ + " of " + folder + " --> Done")

def create_folder(folder):
    if not os.path.isdir(os.path.join("index", folder)):
        os.makedirs(os.path.join("index", folder))

def generateHistogramme_Color(folder):
    create_folder("BGR")
    i=0
    for d in os.listdir(folder):
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
            num_image, _ = path.split(".")
            np.savetxt("index/BGR/"+str(num_image)+".txt" ,feature)
            i+=1
    status_log(generateHistogramme_Color, folder)

def generateHistogramme_HSV(folder):
    create_folder("HSV")
    i = 0

    for d in os.listdir(folder): 
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([img],[0],None,[256],[0,256])
            histS = cv2.calcHist([img],[1],None,[256],[0,256])
            histV = cv2.calcHist([img],[2],None,[256],[0,256])
            feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

            num_image, _ = path.split(".")
            np.savetxt("index/HSV/"+str(num_image)+".txt" ,feature)
            i+=1
    status_log(generateHistogramme_HSV, folder)
   
def generateSIFT(folder):
    create_folder("SIFT")
    i=0
    for d in os.listdir(folder): 
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            featureSum = 0
            sift = cv2.SIFT_create()  
            kps , des = sift.detectAndCompute(img,None)

            num_image, _ = path.split(".")
            if des is not None:
                np.savetxt("index/SIFT/"+str(num_image)+".txt" ,des)
            
                featureSum += len(kps)
            i+=1
    status_log(generateSIFT, folder)   

def generateORB(folder):
    create_folder("ORB")
    i=0
    for d in os.listdir(folder): 
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            orb = cv2.ORB_create()
            key_point1,descrip1 = orb.detectAndCompute(img,None)
    
            num_image, _ = path.split(".")
            if descrip1 is not None:        
                np.savetxt("index/ORB/"+str(num_image)+".txt" ,descrip1)
            i+=1
    status_log(generateORB, folder)   

def generateGLCM(folder):
    create_folder("GLCM")
    distances=[1,-1]
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]
    i=0
    for d in os.listdir(folder): 
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = img_as_ubyte(gray)
            glcmMatrix = graycomatrix(gray, distances=distances, angles=angles,normed=True)
            glcmProperties1 = graycoprops(glcmMatrix,'contrast').ravel()
            glcmProperties2 = graycoprops(glcmMatrix,'dissimilarity').ravel()
            glcmProperties3 = graycoprops(glcmMatrix,'homogeneity').ravel()
            glcmProperties4 = graycoprops(glcmMatrix,'energy').ravel()
            glcmProperties5 = graycoprops(glcmMatrix,'correlation').ravel()
            glcmProperties6 = graycoprops(glcmMatrix,'ASM').ravel()
            feature =np.array([glcmProperties1,glcmProperties2,glcmProperties3,glcmProperties4,glcmProperties5,glcmProperties6]).ravel()
            num_image, _ = path.split(".")
            np.savetxt("index/GLCM/"+str(num_image)+".txt" ,feature)
            i+=1
    status_log(generateGLCM, folder)   
        
def generateLBP(folder):
    create_folder("LBP")
    i=0
    for d in os.listdir(folder): 
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            points=8
            radius=1
            method='default'
            subSize=(70,70)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(350,350))
            fullLBPmatrix = local_binary_pattern(img,points,radius,method)
            histograms = []
            for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
                for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                    subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel()
                    subHist,edges = np.histogram(subVector,bins=int(2**points),range=(0,2**points))
                    histograms = np.concatenate((histograms,subHist),axis=None)
            num_image, _ = path.split(".")
            np.savetxt("index/LBP/"+str(num_image)+".txt" ,histograms)
            i+=1
    status_log(generateLBP, folder)   

def generateHOG(folder):
    create_folder("HOG")
    i=0
    cellSize = (25,25)
    blockSize = (50,50)
    blockStride = (25,25)
    nBins = 9
    winSize = (350,350)
    for d in os.listdir(folder): 
        for path in os.listdir(folder+'/'+d):
            img = cv2.imread(folder+'/'+d+"/"+path)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,winSize)
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
            feature = hog.compute(image)
            num_image, _ = path.split(".")
            np.savetxt("index/HOG/"+str(num_image)+".txt" ,feature)
            i+=1
    status_log(generateHOG, folder)   

if __name__ == "__main__":
    main()