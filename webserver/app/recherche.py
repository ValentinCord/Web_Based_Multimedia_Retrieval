
import os
import cv2
import numpy as np
import math
import pickle as pk
from rmac import RMAC
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from distance import distance_matching
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Dense, Lambda
from keras_preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input #224*224
from keras.applications.xception import Xception, preprocess_input, decode_predictions #299*299
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions #224*224

def extractReqFeatures(fileName, algo_choice):  
    """
    Extract features from an image using the selected algorithm
    Return a vector/matrix of features
    """
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
            w, h, c = img.shape
            new_size = (int(w*0.3), int(h*0.3))
            img = cv2.resize(img, new_size)
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

        elif algo_choice == 'VGG16_false':
            model = VGG16(include_top = False, weights ='imagenet', input_shape = (224, 224, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'VGG16_false_pca':
            model = VGG16(include_top = False, weights ='imagenet', input_shape = (224, 224, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            pca = pk.load(open("static/VGG16_false_pca.pkl",'rb')) 
            pca_data = pca.transform([feature])
            return pca_data

        elif algo_choice == 'VGG16_false_rmac':
            base_model = VGG16(include_top = False, weights ='imagenet', input_shape = (224, 224, 3), pooling = 'avg')
            base_out = base_model.get_layer('block5_conv3').output
            rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)    
            rmac_layer = Lambda(rmac.rmac, input_shape=base_model.output_shape, name="rmac_"+'block5_conv3')
            out = rmac_layer(base_out)
            model = Model(base_model.input, out)
            image = load_img(fileName, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'XCEPTION_false':
            model = Xception(include_top = False, weights ='imagenet', input_shape = (299, 299, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (299, 299))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'XCEPTION_false_pca':
            model = Xception(include_top = False, weights ='imagenet', input_shape = (299, 299, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (299, 299))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            pca = pk.load(open("static/XCEPTION_false_pca.pkl",'rb')) 
            pca_data = pca.transform([feature])
            return pca_data
        
        elif algo_choice == 'XCEPTION_false_rmac':
            base_model = Xception(include_top = False, weights ='imagenet', input_shape = (299, 299, 3), pooling = 'avg')
            base_out = base_model.get_layer('block14_sepconv2').output
            rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)    
            rmac_layer = Lambda(rmac.rmac, input_shape=base_model.output_shape, name="rmac_"+'block14_sepconv2')
            out = rmac_layer(base_out)
            model = Model(base_model.input, out)
            image = load_img(fileName, target_size=(299, 299))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'XCEPTION_true':
            model = Xception(include_top = True, weights ='imagenet', input_shape = (299, 299, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (299, 299))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'XCEPTION_true_pca':
            model = Xception(include_top = True, weights ='imagenet', input_shape = (299, 299, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (299, 299))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            pca = pk.load(open("static/XCEPTION_true_pca.pkl",'rb')) 
            pca_data = pca.transform([feature])
            return pca_data

        elif algo_choice == 'XCEPTION_true_rmac':
            base_model = Xception(include_top = True, weights ='imagenet', input_shape = (299, 299, 3), pooling = 'avg')
            base_out = base_model.get_layer('block14_sepconv2').output
            rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)    
            rmac_layer = Lambda(rmac.rmac, input_shape=base_model.output_shape, name="rmac_"+'block14_sepconv2')
            out = rmac_layer(base_out)
            model = Model(base_model.input, out)
            image = load_img(fileName, target_size=(299, 299))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'MOBILENET_false':
            model = MobileNet(include_top = False, weights ='imagenet', input_shape = (224, 224, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            return feature

        elif algo_choice == 'MOBILENET_false_pca':
            model = MobileNet(include_top = False, weights ='imagenet', input_shape = (224, 224, 3), pooling = 'avg')
            img = load_img(fileName, target_size = (224, 224))
            img = img_to_array(img)
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = preprocess_input(img)
            feature = model.predict(img)
            feature = np.array(feature[0])
            pca = pk.load(open("static/MOBILENET_false_pca.pkl",'rb')) 
            pca_data = pca.transform([feature])
            return pca_data

        elif algo_choice == 'MOBILENET_false_rmac':
            base_model = MobileNet(include_top = False, weights ='imagenet', input_shape = (224, 224, 3), pooling = 'avg')
            base_out = base_model.get_layer('conv_pw_13').output
            rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)    
            rmac_layer = Lambda(rmac.rmac, input_shape=base_model.output_shape, name="rmac_"+'conv_pw_13')
            out = rmac_layer(base_out)
            model = Model(base_model.input, out)
            image = load_img(fileName, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image)
            feature = np.array(feature[0])
            return feature


def recherche(mongo, img_path, descriptors, distance_vector, distance_matrix, cfg):
    """
    Extracts features from input image and compares them with features from database
    Return ordered list of images from database with their distance from input image
    """
    # selecting distance function
    distance_func = distance_matching(cfg, distance_vector, distance_matrix, descriptors)
    
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
                sum_of_distances[desc_name] = dist
            sum_of_distances[desc_name] = max(dist, sum_of_distances[desc_name])

    for img_path, data in distances.items():
        for desc_name, dist in data.items():
            print('before : ', distances[img_path][desc_name])
            distances[img_path][desc_name] /= sum_of_distances[desc_name]
            print('after : ', distances[img_path][desc_name])

    # intersection and correlation are maximized, others are minimized
    # reverse intersection and correlation to minimize them
    for img_path, data in distances.items():
        for desc_name, dist in data.items():
            if desc_name in ['intersection', 'correlation']:
                distances[img_path][desc_name] = 1 - distances[img_path][desc_name]


    # calculating mean normalized distance from each image 
    result = {img_path : round(np.mean(list(data.values())), 3) for img_path, data in distances.items()}

    # sorting image following best distance
    return sorted(result.items(), key = lambda x : x[1], reverse = False) 

def save_metrics(cfg, mongo):
    """
    If the image query is in the database, calculate the metrics and save them in the database
    Metrics are rp_curve, AP(20), AP(50), 20-precision, 50-precision for class and subclass search
    """

    revelant_classe = []
    revelant_subclasse = []

    input_path = cfg['input']['img_path']
    classe_input = input_path.split('/')[2].split('_')[0]
    subclasse_input = input_path.split('/')[2].split('_')[1]

    # check if the result is correctly predicted in terme of class and subclass
    for result in cfg['result']['names']:
        classe = result[0].split('/')[2].split('_')[0]
        subclasse = result[0].split('/')[2].split('_')[1]
        revelant_classe.append(True) if classe_input == classe else revelant_classe.append(False)
        revelant_subclasse.append(True) if subclasse_input == subclasse else revelant_subclasse.append(False)

    # number of images in the database for each class and subclass
    num_class = {
        '0' : 797,
        '1' : 934,
        '2' : 933,
        '3' : 905,
        '4' : 935
    }
    num_subclass = {
        '0' : {
            '0': 155,
            '1': 157,
            '2': 30,
            '3': 145,
            '4': 156,
            '5': 154,
        },
        '4' : {
            '0': 157,
            '1': 156,
            '2': 153,
            '3': 156,
            '4': 157,
            '5': 156,
        },
        '3' : {
            '0': 152,
            '1': 157,
            '2': 128,
            '3': 155,
            '4': 156,
            '5': 157,
        },
        '1' : {
            '0': 155,
            '1': 156,
            '2': 156,
            '3': 156,
            '4': 156,
            '5': 156,
        },
        '2' : {
            '0': 157,
            '1': 155,
            '2': 154,
            '3': 156,
            '4': 155,
            '5': 156,
        }
    }

    recall_class = []
    precision_class = []
    recall_subclass = []
    precision_subclass = []

    # calculating precision and recall
    for i in range(len(revelant_classe)):
        j = i
        val_classe = 0
        val_subclasse = 0   
        while (j >= 0):
            if revelant_classe[j]:
                val_classe += 1
            if revelant_subclasse[j]:
                val_subclasse += 1
            j -= 1

        p_class = val_classe / (i+1)
        r_class = val_classe / num_class[classe_input]
        p_subclass = val_subclasse / (i+1)
        r_subclass = val_subclasse / num_subclass[classe_input][subclasse_input]

        recall_class.append(r_class)
        precision_class.append(p_class)
        recall_subclass.append(r_subclass)
        precision_subclass.append(p_subclass)

    # RP Curve for class retrieval
    plt.plot(recall_class[:50], precision_class[:50])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RP curve for class retrieval")
    save_name = os.path.join("static", "rp_class.png")
    plt.savefig(save_name, format='png', dpi=600)
    plt.close()

    # RP Curve for subclass retrieval
    plt.plot(recall_subclass[:50], precision_subclass[:50])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RP curve for subclass retrieval")
    save_name = os.path.join("static", "rp_subclass.png")
    plt.savefig(save_name, format='png', dpi=600)
    plt.close()

    history = mongo.get_collection('HISTORY')

    # query's class configuration
    data = dict()
    descriptors = [k for k, v in cfg['descriptors'].items() if v == True and k != 'is_selected']
    data['type']            = 'Class'
    data['distance_vect']   = cfg['distance']['vect']
    data['distance_matrix'] = cfg['distance']['matrix']
    data['descriptors']     = "-".join(descriptors)
    data['img_path']        = cfg['input']['img_path'].split("/")[-1]
    data['ap-20']           = round(sum(x for x in precision_class[:20])/20, 3)                
    data['ap-50']           = round(sum(x for x in precision_class[:50])/50, 3)                
    r_class = num_class[classe_input]
    data['r-precision'] = round(sum(revelant_classe[:r_class])/r_class, 3)     
    data['time']            = cfg['result']['time']
    history.insert_one(data)

    # query's subclass configuration
    data = dict()
    descriptors = [k for k, v in cfg['descriptors'].items() if v == True and k != 'is_selected']
    data['type']            = 'Subclass'
    data['distance_vect']   = cfg['distance']['vect']
    data['distance_matrix'] = cfg['distance']['matrix']
    data['descriptors']     = "-".join(descriptors)
    data['img_path']        = cfg['input']['img_path'].split("/")[-1]
    data['ap-20']           = round(sum(x for x in precision_subclass[:20])/20, 3)           
    data['ap-50']           = round(sum(x for x in precision_subclass[:50])/50, 3) 
    r_subclass = num_subclass[classe_input][subclasse_input]
    data['r-precision'] = round(sum(revelant_classe[:r_subclass])/r_subclass, 3)    
    data['time']            = cfg['result']['time']
    history.insert_one(data)

    # saving query's class metrics in cfg to show result
    data = dict()
    data['ap-20']        = sum(x for x in precision_class[:20])/20        
    data['ap-50']        = sum(x for x in precision_class[:50])/50      
    r_class = num_class[classe_input]
    data['r-precision'] = round(sum(revelant_classe[:r_class])/r_class, 3)    
    cfg['metrics']['classe'] = data
    
    # saving query's subclass metrics in cfg to show result
    data = dict()
    data['ap-20']        = sum(x for x in precision_subclass[:20])/20      
    data['ap-50']        = sum(x for x in precision_subclass[:50])/50     
    r_subclass = num_subclass[classe_input][subclasse_input]
    data['r-precision'] = round(sum(revelant_classe[:r_subclass])/r_subclass, 3)  
    cfg['metrics']['subclasse'] = data
