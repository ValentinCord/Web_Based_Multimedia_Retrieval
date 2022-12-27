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
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection VGG16       -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file VGG16.json      --jsonArray \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection XCEPTION    -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file XCEPTION.json  --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection MOBILENET   -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop --type json --file MOBILENET.json --jsonArray  \
&&  mongoimport -h $MONGO_INITDB_HOST --db db --collection HISTORY     -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin --drop