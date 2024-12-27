

from services.redis_client import RedisClient
from models.artwork import Artwork

# from db.mongodb import connect_to_mongodb,insert_document,find_all_documents
import json
from datetime import datetime
from typing import List
import requests
redis_client = RedisClient()
def fetch_artworks_from_api(api_url: str) -> List[Artwork]:
    """
    Gọi API để lấy danh sách artworks và chuyển đổi thành các đối tượng Artwork.
    """
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "artworks" in data:
                return parse_artworks(data["artworks"])
            else:
                raise ValueError("Invalid API response structure.")
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching artworks: {e}")
def get_artworks_from_cache(cache_key: str):
    data = redis_client.get(cache_key)
    if data:
         return parse_artworks(data)
    return None
def save_artworks_to_cache(cache_key: str, artworks: List[Artwork], ttl: int = 30 * 24 * 60 * 60):
    """
    Lưu danh sách artworks vào Redis cache.
    """
    try:
        serialized_data = [
            {
                **artwork.dict(),
                "metadata_create_time": artwork.metadata_create_time.isoformat(),
                "metadata_update_time": artwork.metadata_update_time.isoformat(),
            }
            for artwork in artworks
        ]
        redis_client.set(cache_key, serialized_data, expiration_in_seconds=ttl)
        print(f"✅ Saved {len(artworks)} artworks to cache with key: {cache_key}")
    except Exception as e:
        print(f"❌ Error saving artworks to Redis: {e}")

def parse_artworks(artworks_data: List[dict]) -> List[Artwork]:
    """
    Chuyển đổi danh sách dict từ API hoặc Redis đã được parse sẵn thành danh sách đối tượng Artwork.
    """
    parsed_artworks = []
    for artwork in artworks_data:
        try:
            # Tạo đối tượng Artwork
            parsed_artworks.append(
               Artwork(
                        Artwork_ID=artwork["artID"],
                        Artwork_Title=artwork["title"],
                        Artwork_Tag=artwork["taglist"],
                        Likes_count=artwork["like_count"],
                        Comments_count=artwork["comment_count"],
                    )
            )
        except KeyError as e:
            print(f"❌ Missing field in artwork data: {e}")
        except Exception as e:
            print(f"❌ Error parsing artwork: {artwork}")
            print(f"❌ Error details: {e}")
    return parsed_artworks

