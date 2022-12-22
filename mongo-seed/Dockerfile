FROM mongo

COPY ./data_import /data_import
WORKDIR /data_import

CMD mongoimport --host mongodb --db db --collection BGR  -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file BGR.json  --jsonArray \
&&  mongoimport --host mongodb --db db --collection HOG  -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file HOG.json  --jsonArray \
&&  mongoimport --host mongodb --db db --collection GLCM -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file GLCM.json --jsonArray \
&&  mongoimport --host mongodb --db db --collection HSV  -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file HSV.json  --jsonArray \
&&  mongoimport --host mongodb --db db --collection LBP  -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file LBP.json  --jsonArray \
&&  mongoimport --host mongodb --db db --collection ORB  -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file ORB.json  --jsonArray \
&&  mongoimport --host mongodb --db db --collection SIFT -u 'admin' -p 'admin' --authenticationDatabase admin --drop --type json --file SIFT.json --jsonArray 