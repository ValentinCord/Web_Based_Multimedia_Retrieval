import os

from pymongo import MongoClient

class Mongo():
    def __init__(self):
        self.client = MongoClient(
            os.getenv('MONGO_INITDB_HOST'), 
            int(os.getenv('MONGO_INITDB_PORT')), 
            username = os.getenv('MONGO_INITDB_ROOT_USERNAME'), 
            password = os.getenv('MONGO_INITDB_ROOT_PASSWORD')
        )

        self.db = self.client.db
        self.history = self.db.HISTORY
        self.users = self.db.USERS
        self.users.insert_one({"username": 'admin', "password": 'admin'})

    def get_collection(self, collection_name):
        """
        Return collection object from collection name
        """
        return self.db[collection_name]

    def get_documents(self, collection_name):
        """
        Return documents from collection name (= list of dict)
        """
        collection = self.get_collection(collection_name)
        return [document for document in collection.find()]

