from fastapi import APIRouter
from services.artwork_service import get_artworks_from_cache, fetch_artworks_from_api,save_artworks_to_cache
from services.redis_client import RedisClient
from models.user_action import UserAction
from db.mongodb import MongoDBConnection
from typing import List, Dict
from config import API_URL_GET_ALL_ARTWORK
mongodb_connection = MongoDBConnection()

router = APIRouter()
redis_client = RedisClient()

@router.get("/")
def get_all_artworks():
    cache_key = "all_artworks"

    # Lấy từ Redis cache
    artworks = get_artworks_from_cache(cache_key)

    # Nếu không có cache, gọi API
    if artworks is None:
        print("Cache miss: Fetching from API")
        artworks = fetch_artworks_from_api(API_URL_GET_ALL_ARTWORK)

    return artworks
@router.get("/user-action",response_model=List[UserAction])
def get_all_action():
    db = mongodb_connection.get_db()
    collection = db["user_action"]
    actions = list(collection.find({}))
    for action in actions:
        action["_id"] = str(action["_id"])  # Convert ObjectId to string
    return actions