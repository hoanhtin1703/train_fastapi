# app/services/user_action_service.py
from repositories.user_repository import UserActionRepository
from models.user_action import UserAction
from typing import List, Optional

class UserActionService:
    def __init__(self, user_action_repository: UserActionRepository):
        self.user_action_repository = user_action_repository

    def create_user_action(self, user_action_data: dict) -> str:
        """Tạo một hành động người dùng mới."""
        try:
            user_action = UserAction(**user_action_data)  # Chuyển data thành đối tượng UserAction
            action_id = self.user_action_repository.insert_action(user_action)
            return action_id
        except ValueError as e:
            raise ValueError(f"Error creating user action: {str(e)}")

    def get_all_user_actions(self, limit: int = 10) -> List[dict]:
        """Lấy tất cả các hành động người dùng."""
        return self.user_action_repository.get_all_actions(limit)

    def get_user_actions(self, user_id: str, limit: int = 10) -> List[dict]:
        """Lấy hành động của người dùng theo user_id."""
        return self.user_action_repository.get_actions_by_user(user_id, limit)
