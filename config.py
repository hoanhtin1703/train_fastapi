import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

# Cấu hình chung
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_URL_GET_ALL_ARTWORK = os.getenv("API_URL_GET_ALL_ARTWORK", "http://localhost:7000/api/artwork/all-artwork")
API_URL_GET_ALL_USER_ACTION = os.getenv("API_URL_GET_ALL_USER_ACTION", "http://localhost:7000/api/user/user-actions")
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://ho25428:anh01263654142@tin.1gu7y.mongodb.net/?retryWrites=true&w=majority&appName=Tin")
DATABASE_NAME = os.getenv("DATABASE_NAME", "Art_social")
COLLECTION = os.getenv("COLLECTION", "user_action")



