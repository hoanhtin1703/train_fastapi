import logging
from fastapi import APIRouter, HTTPException
from services.recommendation_service import RecommendationService
from config import API_URL_GET_ALL_ARTWORK, API_URL_GET_ALL_USER_ACTION
import colorlog
from services.redis_client import RedisClient
from typing import List
redis_client = RedisClient()
# Khởi tạo router
router = APIRouter()

# Khởi tạo đối tượng RecommendationService
recommendation_service = RecommendationService()

# Cấu hình logger với colorlog
log_format = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
formatter = colorlog.ColoredFormatter(
    log_format,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# Tạo handler và thiết lập logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Cấu hình logging
logger = logging.getLogger()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Pydantic models để định nghĩa response
from pydantic import BaseModel
from typing import List

class ArtworkRecommendation(BaseModel):
    artID: int
    title: str
    taglist: List[str]
    Likes_count: int
    Comments_count: int
def to_dict(self):
        # Chỉ trả về dictionary chứa các thuộc tính cần thiết
        return {
            "artID": self.artID,
            "title": self.title,
            "taglist": self.taglist,
            "Likes_count": self.Likes_count,
            "Comments_count": self.Comments_count
        }
class RecommendationsResponse(BaseModel):
    recommendations: List[ArtworkRecommendation]


@router.get("/recommend/{user_id}", response_model=RecommendationsResponse)
async def recommend_for_user(user_id: int):
    try:
        n = 10  # Số lượng gợi ý cần lấy
        logger.info(f"Starting recommendation process for user: {user_id}")
        
        # Lấy dữ liệu
        new_artwork = recommendation_service.fetch_artworks_from_cache_or_api(API_URL_GET_ALL_ARTWORK)
        user_action = recommendation_service.fetch_user_action_from_cache_or_api(API_URL_GET_ALL_USER_ACTION)
        artwork, cosine_sim = recommendation_service.preprocess_data(new_artwork)
        # Xây dựng mô hình Collaborative Filtering (CF)
        svd = recommendation_service.build_cf_model(user_action)
        logger.info(f"User action dataset updated successfully.")
        
        # Kiểm tra nếu không có dữ liệu người dùng
        if user_id not in user_action['userID'].values:
            logger.info(f"No interaction data found for User {user_id}. Returning default recommendations.")
            
            # Lấy gợi ý mặc định từ service
            recommend_artwork = recommendation_service.default_recommendation_for_user()

            # Tạo danh sách ArtworkRecommendation từ gợi ý mặc định
            recommendations = [
                ArtworkRecommendation(
                    artID=artwork_data['artID'],  # Trực tiếp sử dụng dữ liệu từ recommend_artwork
                    title=artwork_data['title'],
                    taglist=artwork_data['taglist'],  # Đảm bảo taglist là danh sách
                    Likes_count=artwork_data['Likes_count'],
                    Comments_count=artwork_data['Comments_count']
                )
                for artwork_data in recommend_artwork  # Trực tiếp lấy dữ liệu từ recommend_artwork
            ]
            return RecommendationsResponse(recommendations=recommendations)

        # Lấy các gợi ý hybrid
        artwork_ids = recommendation_service.get_hybrid_recommendations_for_liked(
            user_id=user_id,
            artwork=artwork,
            user_action=user_action,
            svd=svd,
            cosine_sim=cosine_sim,
            n=10
        )

        # Log artworkID cho người dùng
        logger.info(f"Top {n} hybrid recommendations for user {user_id}: {artwork_ids}")  # Chỉ log artworkID

        # Tạo danh sách các ArtworkRecommendation từ danh sách artwork_ids
        recommendations = [
            ArtworkRecommendation(
                artID=int(artwork_id),  # Đảm bảo artID là kiểu int
                title=artwork.loc[artwork['artID'] == artwork_id, 'title'].values[0] if not artwork.loc[artwork['artID'] == artwork_id].empty else "Unknown",
                taglist=artwork.loc[artwork['artID'] == artwork_id, 'taglist'].values[0].split(',') if not artwork.loc[artwork['artID'] == artwork_id].empty else [],
                Likes_count=artwork.loc[artwork['artID'] == artwork_id, 'Likes_count'].values[0],
                Comments_count=artwork.loc[artwork['artID'] == artwork_id, 'Comments_count'].values[0]
            ) 
            for artwork_id in artwork_ids
        ]
        print(recommendations)

        redis_client.set(f"user:{user_id}:recommended_artworks", [rec.dict() for rec in recommendations], expiration_in_seconds=30 * 24 * 60 * 60)
        # Trả về đối tượng RecommendationsResponse chứa danh sách recommendations
        return RecommendationsResponse(recommendations=recommendations)

    except Exception as e:
        # Log lỗi nếu có
        logger.error(f"Error while processing recommendation for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error while processing recommendation for user {user_id}: {str(e)}")
