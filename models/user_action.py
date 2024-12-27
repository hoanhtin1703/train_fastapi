from pydantic import BaseModel
from datetime import datetime
from typing import Literal, Optional

class UserAction(BaseModel):
    userID: int  # ID của người dùng thực hiện hành động
    interaction_type: Literal["like", "comment"]  # Loại hành động
    Artwork_ID: Optional[int] = None  # Dành cho các hành động like, comment, search (ID của đối tượng như artwork_id)

       