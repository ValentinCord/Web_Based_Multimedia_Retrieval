from pymongo import MongoClient

class Mongo():
    def __init__(self):
        self.client = MongoClient('mongodb', 27017, username = 'admin', password = 'admin')
        self.db = self.client.db
    
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

    def init_history(self):
        """
        Initialize history collection
        """
        return self.db.HISTORY

