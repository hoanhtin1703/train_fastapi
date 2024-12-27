from typing import Union
import logging
# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,  # Đảm bảo level là INFO hoặc thấp hơn
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Đảm bảo log được in ra console
)

# Khởi tạo logger
logger = logging.getLogger(__name__)
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    logger.info(f"Received request to recommend for user:")
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}