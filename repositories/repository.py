# repository.py
from pymongo.collection import Collection
from bson import ObjectId
from db.mongodb import MongoDBConnection
from typing import Any, List, Dict

class MongoRepository:
    """Abstract class to define common methods for CRUD operations."""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.db_connection = MongoDBConnection()
        self.collection = self.db_connection.get_db()[collection_name]
    
    def insert_one(self, data: Dict[str, Any]):
        """Insert a single document into the collection."""
        return self.collection.insert_one(data).inserted_id
    
    def insert_many(self, data: List[Dict[str, Any]]):
        """Insert multiple documents into the collection."""
        return self.collection.insert_many(data).inserted_ids
    
    def find_one(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Find a single document based on query."""
        return self.collection.find_one(query)
    
    def find_many(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find multiple documents based on query."""
        query = query or {}
        return list(self.collection.find(query))
    
    def update_one(self, query: Dict[str, Any], update: Dict[str, Any]):
        """Update a single document based on query."""
        return self.collection.update_one(query, {"$set": update})
    
    def delete_one(self, query: Dict[str, Any]):
        """Delete a single document based on query."""
        return self.collection.delete_one(query)

    def delete_many(self, query: Dict[str, Any]):
        """Delete multiple documents based on query."""
        return self.collection.delete_many(query)


class UserRepository(MongoRepository):
    """Repository class for User entity."""

    def __init__(self):
        super().__init__("users")  # Assume the collection is named "users"
    
    def find_user_by_id(self, user_id: str):
        """Find user by user_id."""
        return self.find_one({"_id": ObjectId(user_id)})

    def create_user(self, user_data: Dict[str, Any]):
        """Create a new user."""
        return self.insert_one(user_data)
