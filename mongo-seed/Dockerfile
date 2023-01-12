ARG MONGO_INITDB_ROOT_USERNAME
ARG MONGO_INITDB_ROOT_PASSWORD
ARG MONGO_INITDB_HOST

FROM mongo

COPY ./data_import /data_import
WORKDIR /data_import

CMD mongoimport -h $MONGO_INITDB_HOST --db db --collection BGR         -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file BGR.json       --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection HOG         -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file HOG.json       --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection GLCM        -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file GLCM.json      --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection HSV         -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file HSV.json       --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection LBP         -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file LBP.json       --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection ORB         -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file ORB.json       --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection SIFT        -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file SIFT.json      --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection VGG16_false       -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file VGG16_false.json      --jsonArray \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION_false    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION_false.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION_true    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION_true.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection MOBILENET_false   -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file MOBILENET_false.json --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection VGG16_false_pca       -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file VGG16_false_pca.json      --jsonArray \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION_false_pca    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION_false_pca.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION_true_pca    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION_true_pca.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection MOBILENET_false_pca   -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file MOBILENET_false_pca.json --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection VGG16_false_rmac       -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file VGG16_false_rmac.json      --jsonArray \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION_false_rmac    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION_false_rmac.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION_true_rmac    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION_true_rmac.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection MOBILENET_false_rmac   -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file MOBILENET_false_rmac.json --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection HISTORY     -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop