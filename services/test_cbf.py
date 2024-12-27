import numpy as np
from datetime import datetime
from typing import List, Dict
import logging
import requests
import joblib
import pandas as pd
# from db.mongodb import MongoDBConnection

from config import API_URL_GET_ALL_ARTWORK
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import time
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

# Configure Redis client and logging
# Create a custom logger
# Cấu hình logger
import colorlog

# Cấu hình logger với colorlog
log_format = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
formatter = colorlog.ColoredFormatter(
    log_format,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',  # Màu xanh lá cho INFO
        'WARNING': 'yellow',
        'ERROR': 'red',  # Màu đỏ cho ERROR
        'CRITICAL': 'bold_red',
    }
)

# Tạo handler và thiết lập logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Cấu hình logging
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

import redis
import json

class RedisClient:
    def __init__(self):
        self.client = redis.StrictRedis(host='localhost', port=6379, db=0)

    def get(self, key):
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def set(self, key, value):
        self.client.set(key, json.dumps(value))

class RecommendationService:
    def __init__(self):
        pass
    def load_existing_artworks(self) -> pd.DataFrame:
        """
        Load the existing artworks from CSV if available.
        If the file doesn't exist, return an empty DataFrame with the required columns.
        """
        try:
            existing_artworks = pd.read_csv('./dataset/artstation_main_data_with_ids.csv')
        except FileNotFoundError:
            print("Existing data not found. Initializing new dataset.")
            existing_artworks = pd.DataFrame(columns=['Artwork_ID', 'Artwork_Title', 'Artwork_Tag', 'Likes_count', 'Comments_count'])
        return existing_artworks
    def process_data(self,api_url:str) -> pd.DataFrame:
        """
        Fetch artworks from Redis cache if available, otherwise fetch from API.
        Then, combine the new data with the existing dataset.
        """
        # 1. Get artworks from cache
        cache_key = "all_artworks"
        new_artworks = self.get_artworks_from_cache(cache_key)
        if not new_artworks:
            print("Fetching artworks from API.")
            new_artworks = self.fetch_artworks_from_api(api_url)
        else:
            print(f"Fetched {len(new_artworks)} artworks from cache.")
        
        # 2. Convert new artworks into DataFrame
        columns_artwork = ['Artwork_ID', 'Artwork_Title', 'Artwork_Tag', 'Likes_count', 'Comments_count']
        new_artworks_df = pd.DataFrame(new_artworks,columns=columns_artwork)
        new_artworks_df.to_csv("parse_artworks.csv",index=False)
        # 3. Load existing dataset (artworks)
        existing_artworks = self.load_existing_artworks()

        # 4. Combine the old and new artworks and drop duplicates
        combined_artworks = pd.concat([existing_artworks, new_artworks_df]).drop_duplicates(subset='Artwork_ID', keep='last')
        # Đọc dữ liệu và chuyển cột 'Artwork_ID' thành integer
        combined_artworks['Artwork_ID'] = combined_artworks['Artwork_ID'].astype(int)

        # 5. Save the updated dataset back to disk
        combined_artworks.to_csv('./dataset/artstation_main_data_with_ids.csv', index=False)
        print("Artwork data updated successfully.")
        combined_artworks['Artwork_Tag'] = combined_artworks['Artwork_Tag'].apply(self.process_tags)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(combined_artworks['Artwork_Tag'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logger.info(f"User combined_artworks data updated successfully:\n{combined_artworks.head()}")
        return combined_artworks, cosine_sim
    def fetch_artworks_from_api(self, api_url: str) -> List[dict]:
        """
        Fetch artworks from the API and parse the response.
        """
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "artworks" in data:

                return self.parse_artworks(data["artworks"])
            else:
                raise ValueError("Invalid API response structure. 'success' or 'artworks' field missing.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching artworks from API: {e}")
            raise RuntimeError(f"Error fetching artworks from API: {e}")
        except ValueError as e:
            logger.error(f"Data parsing error: {e}")
            raise ValueError(f"Data parsing error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}")

    def parse_artworks(self, artworks_data: List[dict]) -> List[dict]:
        parsed_artworks = []
        for artwork in artworks_data:
            try:
                parsed_artworks.append({
                    "Artwork_ID": artwork["artID"],
                    "Artwork_Title": artwork["title"],
                    "Artwork_Tag": artwork["taglist"],
                    "Likes_count": artwork["like_count"],
                    "Comments_count": artwork["comment_count"],
                })
            except KeyError as e:
                logger.warning(f"Missing field in artwork data: {e}")
        return parsed_artworks


    def get_artworks_from_cache(self, cache_key: str) -> List[dict]:
        """
        Fetch artworks from Redis cache if available, and parse them using `parse_artworks`.
        """
        data = redis_client.get(cache_key)  # Lấy dữ liệu từ Redis
        if data:
            try:
                # If data is a JSON string, parse it into a Python object
                # Call parse_artworks to process the data
                parsed_artworks = self.parse_artworks(data)
                if parsed_artworks:
                    logger.info(f"Successfully parsed {len(parsed_artworks)} artworks from cache.")
                    return parsed_artworks

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding cached artwork data: {e}")
            except Exception as e:
                logger.error(f"Unexpected error when processing cache data: {e}")
        else:
            logger.info("No data found in cache.")

        return None  # Return an empty list if the cache is empty or invalid
    def main():
        # URL của API để lấy dữ liệu artwork (cần thay thế bằng URL thực tế của bạn)
        api_url = API_URL_GET_ALL_ARTWORK

        # Khởi tạo đối tượng RecommendationService
        recommendation_service = RecommendationService()

        try:
            # 1. Tiến hành xử lý dữ liệu artwork từ API hoặc Cache
            print("Processing artworks data...")
            combined_artworks, cosine_sim = recommendation_service.process_data(api_url)
            print(f"Processed {len(combined_artworks)} artworks data.")
            
            # # 2. Đưa ra các gợi ý cho một người dùng (ví dụ: user_id = 113)
            # user_id = 113  # Bạn có thể thay đổi ID người dùng
            # artwork_id = 123  # ID của tác phẩm bạn muốn tìm gợi ý cho người dùng

            # # Gọi hàm gợi ý (hybrid hoặc content-based collaborative filtering)
            # print(f"Getting recommendations for User {user_id} and Artwork {artwork_id}...")
            # hybrid_top_n = recommendation_service.hybrid_recommendation_with_strong_interactions(
            #     user_id, recommendation_service.svd, cosine_sim, combined_artworks, recommendation_service.db, n=10
            # )

            # # In kết quả gợi ý
            # print(f"Top 10 Recommendations for User {user_id}:")
            # for idx, (artwork_id, score) in enumerate(hybrid_top_n, 1):
            #     print(f"{idx}. Artwork ID: {artwork_id}, Predicted Score: {score:.4f}")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(f"An error occurred: {e}")

    if __name__ == "__main__":
        main()
