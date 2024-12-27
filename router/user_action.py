from fastapi import APIRouter,HTTPException
import logging
import json
from services.user_action_service import UserActionService
from models.user_action import UserAction
from db.mongodb import MongoDBConnection
from typing import List, Dict
mongodb_connection = MongoDBConnection()
logging.basicConfig(
    level=logging.INFO,  # Đảm bảo level là INFO hoặc thấp hơn
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Đảm bảo log được in ra console
)

# Khởi tạo logger
logger = logging.getLogger(__name__)

router = APIRouter()

def convert_objectid_to_str(actions):
    for action in actions:
        action["_id"] = str(action["_id"])  # Chuyển ObjectId thành chuỗi
    return actions
@router.get("/", response_model=List[Dict])
def get_all_actions():
    db = mongodb_connection.get_db()  # Lấy kết nối tới database
    collection = db["user_action"]
    actions = list(collection.find({}))  # Chuyển con trỏ thành danh sách
    return convert_objectid_to_str(actions) 
@router.get("/user/{user_id}", response_model=List[Dict])
def get_user_actions(user_id: int):
    db = mongodb_connection.get_db()  # Lấy kết nối tới database
    collection = db["user_action"]
    actions = list(collection.find({"user_id": user_id}))  # Chuyển con trỏ thành danh sách
    return convert_objectid_to_str(actions)




@router.post("/new")
async def receive_user_action(new_user_action: UserAction):
    logger.info("user_action", new_user_action)
    db = mongodb_connection.get_db()
    collection = db["user_action"]
    try:
        if isinstance(new_user_action, dict):
            collection.insert_one(new_user_action)
        else:
            # Nếu không phải dict, chuyển đổi thành dict
            collection.insert_one(new_user_action.__dict__)
        return("Add user_action successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving user action: {e}")
   
    


