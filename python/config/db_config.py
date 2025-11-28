import pymongo
from config.loader import config

# MongoDB configuration
MONGO_HOST = config['database']['host']
MONGO_PORT = config['database']['port']
MONGO_DB_NAME = config['database']['name']

def get_db_connection():
    """
    Create and return a MongoDB database connection.
    
    Returns:
        db: MongoDB database object
    """
    client = pymongo.MongoClient(config['database']['uri'])
    db = client[MONGO_DB_NAME]
    return db


def get_collection(collection_name):
    """
    Get a specific collection from the database.
    
    Args:
        collection_name: name of the collection
        
    Returns:
        collection: MongoDB collection object
    """
    db = get_db_connection()
    return db[collection_name]
