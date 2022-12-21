mongoimport --db db --collection BGR --drop --file /docker-entrypoint-initdb.d/data_import/BGR.json --jsonArray
mongoimport --db db --collection HSV --drop --file /docker-entrypoint-initdb.d/data_import/HSV.json --jsonArray
mongoimport --db db --collection LBP --drop --file /docker-entrypoint-initdb.d/data_import/LBP.json --jsonArray
mongoimport --db db --collection ORB --drop --file /docker-entrypoint-initdb.d/data_import/ORB.json --jsonArray
mongoimport --db db --collection SIFT --drop --file /docker-entrypoint-initdb.d/data_import/SIFT.json --jsonArray
mongoimport --db db --collection GLCM --drop --file /docker-entrypoint-initdb.d/data_import/GLCM.json --jsonArray
mongoimport --db db --collection HOG --drop --file /docker-entrypoint-initdb.d/data_import/HOG.json --jsonArray