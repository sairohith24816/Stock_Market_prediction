import pymongo

# MongoDB configuration
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB_NAME = "stocks"

def get_db_connection():
    """
    Create and return a MongoDB database connection.
    
    Returns:
        db: MongoDB database object
    """
    client = pymongo.MongoClient(f"mongodb://{MONGO_HOST}:{MONGO_PORT}/")
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
