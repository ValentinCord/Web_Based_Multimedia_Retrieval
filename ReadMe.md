# **MIR Project**

## **What is a MIR Project ?**

MIR stands for Multimedia Information Retrieval. 
Since the digital content explosion, the size of 
multimedia databases has been growing steadily.
A MIR project aims to make the retrieval of revelant 
content easier and faster for the user.

A common example is Google Image: based on a text 
description, the search engine selects relevant images.
Another way to achieve this search is to use an image in input.
In this project, the MIR system aims to find images similar 
to a query image based on visual features extraction.

## **How does it work ?**

The MIR system is be based on the following steps:
- Index all the images in the database according to a set of descriptors.
- Index the query image according to the same descriptors.
- Calculate the distance between the query image and the others to define a similarity order.

The **descriptors** are the features that are extracted from the images. 
In this context, three types of descriptors are used: 
- Color descriptors:
    - RGB histogram
    - HSV histogram
- Shape descriptors:
    - ORB: Oriented Fast and Rotated Brief
    - HOG: Histogram of Oriented Gradients
    - SIFT: Scale-Invariant Feature Transform
    - SURF: Speeded Up Robust Features
- Texture descriptors
    - GLCM: Gray-Level Co-Occurrence Matrix
    - LBP: Local Binary Patterns
- Deep Learning
    - VGG16
    - Xception      
    - MobileNet

The **distance** between two descriptors can be calculated using different methods:
- Vector distance:
    - Euclidean
    - Correlation
    - Interesection
    - Chi-Square
    - Bhattcharyya  
- Matrix distance:  
    - Brute Force Matcher
    - Flann

At the end, the final distance between two images is the average 
of the normalized distances between their descriptors.

## **How to use it ?**

### **Docker Installation on Ubuntu**
```console
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo 
$ apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/
$ linux/ubuntu $(lsb_release -cs) stable"
$ sudo apt-get update
$ sudo apt-get install -y docker-ce
$ sudo systemctl status docker
$ docker --version
```

### **Docker Compose Installation on Ubuntu**
```console
$ sudo curl -L "https://github.com/docker/compose/rel... -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
$ docker-compose --version
```

### **Docker Compose Utilization**
```console
$ docker-compose build
$ docker-compose up
$ docker-compose down
```

---

## **Authors**

Corduant Valentin, FPMs (Be) : valentin.corduant@umons.ac.be

Vansnick Tanguy, FPMs (Be) : tanguy.vansnick@umons.ac.be
