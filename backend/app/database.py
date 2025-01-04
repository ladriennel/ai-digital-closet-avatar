from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId


client = MongoClient('mongodb://localhost:27017/')
db = client.digicloset  # This creates/uses a database named 'digicloset'

def add_clothing_items(detected_items: list) -> list:
    """
    Add detected clothing items to MongoDB
    Returns: List of items with MongoDB IDs
    """
    # Add timestamp to each item
    for item in detected_items:
        item['created_at'] = datetime.now(timezone.utc)
    
    # Insert into MongoDB
    result = db.clothes.insert_many(detected_items)
    
    # Add MongoDB IDs to the items
    for item, item_id in zip(detected_items, result.inserted_ids):
        item['_id'] = str(item_id)
    
    return detected_items

def get_all_clothes():
    """Get all clothing items"""
    clothes = list(db.clothes.find())
    # Convert ObjectId to string for JSON serialization
    for item in clothes:
        item['_id'] = str(item['_id'])
    return clothes

def get_clothes_by_category(category: str):
    """Get clothes by category"""
    clothes = list(db.clothes.find({'classification.category': category}))
    for item in clothes:
        item['_id'] = str(item['_id'])
    return clothes