
import os
import cv2
import numpy as np
import math

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

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

def extractReqFeatures(fileName, algo_choice):  

    if fileName : 
        img = cv2.imread(fileName)
            
        if algo_choice == 'BGR':
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            return np.concatenate((histB, np.concatenate((histG,histR),axis=None)), axis=None)
        
        elif algo_choice == 'HSV':
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            return np.concatenate((histH, np.concatenate((histS,histV),axis=None)), axis=None)

        elif algo_choice == 'SIFT': 
            sift = cv2.SIFT_create()
            kps , vect_features = sift.detectAndCompute(img,None)
            return vect_features

        elif algo_choice == 'ORB': 
            orb = cv2.ORB_create()
            key_point1, vect_features = orb.detectAndCompute(img,None)
            return vect_features

        elif algo_choice == 'GLCM': 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            glcm = graycomatrix(img, distances=[1, -1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True)
            glcmProperties1 = graycoprops(glcm,'contrast').ravel()
            glcmProperties2 = graycoprops(glcm,'dissimilarity').ravel()
            glcmProperties3 = graycoprops(glcm,'homogeneity').ravel()
            glcmProperties4 = graycoprops(glcm,'energy').ravel()
            glcmProperties5 = graycoprops(glcm,'correlation').ravel()
            glcmProperties6 = graycoprops(glcm,'ASM').ravel()
            return np.array([glcmProperties1,
                                        glcmProperties2,
                                        glcmProperties3,
                                        glcmProperties4,
                                        glcmProperties5,
                                        glcmProperties6]).ravel()
        
        elif algo_choice == 'LBP':
            points=8
            radius=1
            method='default'
            subSize=(70,70)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(350,350))
            fullLBPmatrix = local_binary_pattern(img,points,radius,method)
            hist = []
            for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
                for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                    subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel()
                    subHist,edges = np.histogram(subVector,bins=int(2**points),range=(0,2**points))
                    hist = np.concatenate((hist, subHist), axis=None)
            return hist

        elif algo_choice == 'HOG':
            cellSize = (25,25)
            blockSize = (50,50)
            blockStride = (25,25)
            nBins = 9
            winSize = (350,350)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,winSize)
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
            return hog.compute(image)

def recherche(mongo, img_path, descriptors, distance_vector, distance_matrix, cfg):

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
    elif distance_matrix == 'flann':
        distance_matrix_func = flann
    elif distance_matrix == 'bruteForceMatching':
        distance_matrix_func = bruteForceMatching

    # matching of descriptors with right distance
    distance_func = dict()
    for desc_name in descriptors:
        distance_func[desc_name] = distance_vector_func if desc_name in cfg['vector'] else distance_matrix_func
    
    # indexing input image
    features_input = {desc : extractReqFeatures(img_path, desc) for desc in descriptors}

    # loading features of database from mongo
    features = dict()
    for desc_name in descriptors:
        documents = mongo.get_documents(desc_name)
        features[desc_name] = {}
        for document in documents:
            for k, v in document.items():
                if k != '_id':
                    feature = np.asarray(v)
                    img_path = os.path.join('static', 'db', k + '.jpg')
                    features[desc_name][img_path] = feature

    # calculating distance from each image for each descriptor
    distances = dict()
    for desc_name, data in features.items():
        for img_path, feature in data.items():   
            if img_path not in distances:
                distances[img_path] = {}         
            distances[img_path][desc_name] = distance_func[desc_name](features_input[desc_name], feature)
    
    # normalizing distance
    sum_of_distances = dict()
    for img_path, data in distances.items():
        for desc_name, dist in data.items():
            if desc_name not in sum_of_distances:
                sum_of_distances[desc_name] = 0
            sum_of_distances[desc_name] += dist
    for img_path, data in distances.items():
        for desc_name, dist in data.items():
            dist /= sum_of_distances[desc_name]

    # calculating mean normalized distance from each image 
    result = {img_path : np.mean(list(data.values())) for img_path, data in distances.items()}

    # sorting image following best distance
    if distance_vector in ["correlation", "intersection"]:
        return sorted(result.items(), key = lambda x : x[1], reverse = True) 
    else:
        return sorted(result.items(), key = lambda x : x[1], reverse = False) 

def rappel_precision():

    rappel_precision = []
    rappels = []
    precisions = []

    filename_req = os.path.basename(fileName)
    num_image, _ = filename_req.split(".")
    classe_image_requete = int(num_image)/100
    val = 0

    for j in range(sortie):
        classe_image_proche=(int(nom_image_plus_proches[j].split('.')[0]))/100
        classe_image_requete = int(classe_image_requete)
        classe_image_proche = int(classe_image_proche)

        if classe_image_requete == classe_image_proche:
            rappel_precision.append(True) #Bonne classe (pertinant)
            val += 1
        else:
            rappel_precision.append(False) #Mauvaise classe (non pertinant)

    print(rappel_precision)
    print(sortie)

    for i in range(sortie):
        j = i
        val = 0
        while(j >= 0):
            if rappel_precision[j]:
                val += 1
            j -= 1

        precision = val/(i+1)
        rappel = val/sortie

        rappels.append(rappel)
        precisions.append(precision)

    #Création de la courbe R/P
    plt.plot(rappels,precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("R/P"+str(sortie)+" voisins de l'image n°"+num_image)

    #Enregistrement de la courbe RP
    save_folder=os.path.join(".",num_image)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_name=os.path.join(save_folder,num_image+'.png')
    plt.savefig(save_name,format='png',dpi=600)
    plt.close()

    #Affichage de la courbe R/P
    img = cv2.imread(save_name,1) #load image in color

