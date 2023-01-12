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

At the end, the final distance between two images is the average 
of the normalized distances between their descriptors.

## **How to use it ?**

### **Docker Compose Utilization**
```console
$ docker-compose build
$ docker-compose up
$ docker-compose down
```

### **Push to DockerHub**
~~~
docker login -u valentincorduant

docker tag fde36340276f valentincorduant/seeder:v2.0
docker tag 7da581450255 valentincorduant/webserver:v2.0

docker push valentincorduant/seeder:v2.0
docker push valentincorduant/webserver:v2.0
~~~

---

## **Authors**

Corduant Valentin, FPMs (Be) : valentin.corduant@umons.ac.be

Vansnick Tanguy, FPMs (Be) : tanguy.vansnick@umons.ac.be
