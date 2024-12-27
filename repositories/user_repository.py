# app/repositories/user_action_repository.py
from pymongo.collection import Collection
from typing import List, Optional
from datetime import datetime
from models.user_action import UserAction

class UserActionRepository:
    def __init__(self, collection: Collection):
        self.collection = collection

    def insert_action(self, action: UserAction) -> str:
        """Thêm một hành động người dùng vào MongoDB."""
        action_dict = action.dict()  # Chuyển object Pydantic thành dict
        action_dict["timestamp"] = datetime.now()  # Đảm bảo rằng timestamp là hiện tại
        result = self.collection.insert_one(action_dict)
        return str(result.inserted_id)  # Trả về id của document vừa thêm

    def get_all_actions(self, limit: int = 10) -> List[dict]:
        """Lấy tất cả các hành động người dùng."""
        actions_cursor = self.collection.find({}).limit(limit)
        actions = [action for action in actions_cursor]
        return actions

    def get_actions_by_user(self, user_id: str, limit: int = 10) -> List[dict]:
        """Lấy hành động của người dùng theo user_id."""
        actions_cursor = self.collection.find({"user_id": user_id}).limit(limit)
        actions = [action for action in actions_cursor]
        return actions
