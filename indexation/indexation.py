from functions import generateGLCM, generateHistogramme_Color, generateHistogramme_HSV, generateHOG, generateLBP, generateORB, generateSIFT
from time import time

# Génération des tous les descripteurs + benchmark pour le rapport
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

    bgr = e1-s1
    hsv = e2-s2
    hog = e3-s3 
    lbp = e4-s4 
    orb = e5-s5 
    sift = e6-s6
    glcm = e7-s7

    with open("benchmark.txt", "w") as fin:
        fin.write('Temps pour calculer les descripteurs :' + '\n')
        fin.write(f'BGR : {bgr} s' + '\n')
        fin.write(f'HSV : {hsv} s'+ '\n')
        fin.write(f'HOG : {hog} s'+ '\n')
        fin.write(f'LBP : {lbp} s'+ '\n')
        fin.write(f'ORB : {orb} s'+ '\n')
        fin.write(f'SIFT : {sift} s'+ '\n')
        fin.write(f'GLCM : {glcm} s'+ '\n')

if __name__ == "__main__":
    main()