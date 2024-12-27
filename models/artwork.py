from pydantic import BaseModel
from typing import List, Optional
class Artwork(BaseModel):
    Artwork_ID: int
    Artwork_Title: str
    Artwork_Tag: List[str]  # art là một mảng các chuỗi
    Comments_count: int
    Likes_count: int

    def dict(self, *args, **kwargs):
        data = super().dict(*args, **kwargs)
        # Chuyển datetime thành ISO 8601 string
        return data
