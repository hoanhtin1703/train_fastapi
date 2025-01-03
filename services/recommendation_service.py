import numpy as np
import pandas as pd
import logging
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.redis_client import RedisClient
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from config import API_URL_GET_ALL_ARTWORK
import requests
from typing import List
from surprise import accuracy
import time
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import ast

# Initialize Redis client
redis_client = RedisClient()

class RecommendationService:
    def __init__(self):
        pass
    # Artwork Data
    def fetch_artworks_from_cache_or_api(self, api_url: str) -> pd.DataFrame:
        cache_key = "all_artworks"
        new_artworks = self.get_artworks_from_cache(cache_key)
        
        if not new_artworks:
            logger.info("Fetching artworks from API.")
            new_artworks = self.fetch_artworks_from_api(api_url)
        else:
            logger.info(f"Fetched {len(new_artworks)} artworks from cache.")
        
        # Convert new artworks to DataFrame
        columns_artwork = ['artID', 'title', 'taglist', 'Likes_count', 'Comments_count']
        new_artworks_df = pd.DataFrame(new_artworks, columns=columns_artwork)
        new_artworks_df.to_csv("new_artworks.csv",index=False)
        logger.info("New artwork fetched successfully", new_artworks_df)
        # Load existing artworks and combine
        return new_artworks_df

    def fetch_artworks_from_api(self, api_url: str) -> List[dict]:
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "artworks" in data:
                return self.parse_artworks(data["artworks"])
            else:
                raise ValueError("Invalid API response structure.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching artworks from API: {e}")
            raise RuntimeError(f"Error fetching artworks from API: {e}")
        except ValueError as e:
            logger.error(f"Data parsing error: {e}")
            raise ValueError(f"Data parsing error: {e}")

    def parse_artworks(self, artworks_data: List[dict]) -> List[dict]:
        parsed_artworks = []
        for artwork in artworks_data:
            try:
                parsed_artworks.append({
                    "artID": artwork["artID"],
                    "title": artwork["title"],
                    "taglist": artwork["taglist"],
                    "Likes_count": artwork["like_count"],
                    "Comments_count": artwork["comment_count"],
                })
            except KeyError as e:
                logger.warning(f"Missing field in artwork data: {e}")
        return parsed_artworks

    def get_artworks_from_cache(self, cache_key: str) -> List[dict]:
        data = redis_client.get(cache_key)
        if data:
            try:
                parsed_artworks = self.parse_artworks(data)
                if parsed_artworks:
                    logger.info(f"Successfully parsed {len(parsed_artworks)} artworks from cache.")
                    return parsed_artworks
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding cached artwork data: {e}")
        return None    

    # User Action data
    def fetch_user_action_from_cache_or_api(self,api_url:str) -> pd.DataFrame:
        cacheKey = "user_actions"
        new_user_actions = self.fetch_user_action_from_cache(cacheKey)
        if not new_user_actions:
            logger.info("Fetching user actions from API.")
            new_user_actions = self.fetch_user_action_from_api(api_url)


        # Define the columns for user actions DataFrame
        columns_user_action = ['userID', 'artID', 'interaction_type']

        # Log new user actions
        logger.info("User actions fetched successfully", new_user_actions)

        # Convert new user actions to DataFrame
        new_user_actions_df = pd.DataFrame(new_user_actions, columns=columns_user_action)
        new_user_actions_df.to_csv("new_user_actions.csv",index=False)
        # Ensure correct data types
        new_user_actions_df['userID'] = new_user_actions_df['userID'].astype(int)
        new_user_actions_df['artID'] = new_user_actions_df['artID'].astype(int)

        return new_user_actions_df

    def fetch_user_action_from_api(self,api_url:str) -> List[dict]:
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get("success") and "user_actions" in data:
                return self.parse_user_actions(data["data"])
            else:
                raise ValueError("Invalid API response structure.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching user actions from API: {e}")
            raise RuntimeError(f"Error fetching user actions from API: {e}")
        except ValueError as e:
            logger.error(f"Data parsing error: {e}")
            raise ValueError(f"Data parsing error: {e}")
    def fetch_user_action_from_cache(self,cache_key:str) -> List[dict]:
        data = redis_client.get(cache_key)
        if data:
            try:
                parsed_user_actions = self.parse_user_actions(data)
                if parsed_user_actions:
                    logger.info(f"Successfully parsed {len(parsed_user_actions)} user actions from cache.")
                    return parsed_user_actions
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding cached user action data: {e}")
        return None
    def parse_user_actions(self,user_action_data : List[dict])->List[dict]:    
        parsed_action = []
        for action in user_action_data:
            try:
                parsed_action.append({
                    "userID": action["userID"],
                    "artID": action["artID"],
                    "interaction_type": action["interaction_type"]
                })
            except KeyError as e:
                logger.warning(f"Missing field in artwork data: {e}")
        return parsed_action
    
    # Kiểm tra nếu tag là chuỗi hợp lệ
    def process_tags(self, tag):
        """Xử lý và chuyển các thẻ thành chuỗi từ"""
        try:
            # Kiểm tra xem tag có phải là list hay không
            if isinstance(tag, list):
                # Nếu là list, nối các phần tử thành chuỗi và xử lý các từ ghép
                processed_tag = ' '.join([word.replace(' ', '') if len(word.split()) > 1 else word for word in tag])
                return processed_tag
            elif isinstance(tag, str) and tag != '':
                # Nếu là chuỗi, xử lý các từ ghép có dấu cách
                processed_tag = ''.join([word.replace(' ', '') if len(word.split()) > 1 else word for word in tag.split()])
                return processed_tag
            return ''
        except Exception as e:
            logger.error(f"Error processing tag: {e}")
            return ''


    def preprocess_data(self, artwork):
        """Tiền xử lý dữ liệu từ CSV hoặc nguồn khác"""
        
        # Kiểm tra dữ liệu trong 'taglist' trước khi xử lý
        logger.info(f"Data before preprocessing: {artwork['taglist'].head()}")
        
        # In dữ liệu để kiểm tra kiểu dữ liệu và giá trị trong 'taglist'
        print("Before preprocessing:")
        print(artwork['taglist'].head())  # In 5 giá trị đầu tiên của cột 'taglist'

        # Tiến hành xử lý taglist
        artwork['taglist'] = artwork['taglist'].apply(lambda x: self.process_tags(x) if isinstance(x, (str, list)) else '')
        
        # In dữ liệu sau khi tiền xử lý
        print("After preprocessing:")
        print(artwork['taglist'].head())  # In 5 giá trị đầu tiên sau khi xử lý

        logger.info(f'After preprocessing data for artwork: {artwork["taglist"]}')
        logger.info(f"Data after preprocessing: {artwork['taglist'].head()}")

        # Sử dụng TfidfVectorizer để tạo ma trận TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(artwork['taglist'])
        
        # Tính độ tương đồng cosine giữa các sản phẩm
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        logger.info('Preprocessing completed successfully.')
        
        return artwork, cosine_sim

    def get_content_based_recommendations_for_liked(self,user_id,user_action,artwork, cosine_sim, n=10):
        """Gợi ý các tác phẩm có độ tương đồng với các artwork mà người dùng đã like"""
        # Lọc các tác phẩm mà người dùng đã like
        liked_artworks = user_action[(user_action['userID'] == user_id) & (user_action['interaction_type'] == 1)]['artID'].values
        all_artworks = artwork['artID'].values  # Lấy danh sách tất cả các tác phẩm có trong artwork

        # Tính toán điểm tương đồng cho tất cả các tác phẩm trong cơ sở dữ liệu
        similar_scores = np.zeros(len(all_artworks))  # Khởi tạo một mảng để lưu điểm tương đồng

        # Lặp qua tất cả các tác phẩm mà người dùng đã like
        for artwork_id in liked_artworks:
            idx = artwork[artwork['artID'] == artwork_id].index[0]  # Tìm chỉ số của tác phẩm trong artwork
            sim_scores = cosine_sim[idx]  # Lấy độ tương đồng của tác phẩm với tất cả các tác phẩm khác
            similar_scores += sim_scores  # Cộng dồn các điểm tương đồng

        # Lọc các tác phẩm chưa được like
        artworks_to_predict = [artwork_id for artwork_id in all_artworks if artwork_id not in liked_artworks]

        # Lấy chỉ số của các tác phẩm chưa được like, sắp xếp theo độ tương đồng từ cao đến thấp
        artworks_to_predict_sorted = sorted(artworks_to_predict, key=lambda x: similar_scores[np.where(all_artworks == x)[0][0]], reverse=True)

        # Lấy top n tác phẩm chưa được like
        top_similar_indices = artworks_to_predict_sorted[:n]

        # Log các tác phẩm tương đồng nhất
        print(f"Top {n} recommended artworks for User {user_id} based on liked artworks:")
        for i, artwork_id in enumerate(top_similar_indices, 1):
            print(f"{i}. Artwork ID: {artwork_id}, Similarity Score: {similar_scores[np.where(all_artworks == artwork_id)[0][0]]}")

        return top_similar_indices, similar_scores

# user_action['interaction_type']=user_action['interaction_type'].map({'like': 1, 'comment': 0})


# 2. Collaborative Filtering (CF): Sử dụng SVD để dự đoán sự yêu thích của người dùng
    def build_cf_model(self,user_action, test_size=0.2, n_factors=150, lr_all=0.002, reg_all=0.02):
        # Kiểm tra dữ liệu đầu vào
        if user_action.empty:
            raise ValueError("Dữ liệu đầu vào trống. Vui lòng cung cấp dữ liệu hợp lệ.")

        if not all(col in user_action.columns for col in ['userID', 'artID', 'interaction_type']):
            raise ValueError("Dữ liệu phải chứa các cột 'userID', 'artID', và 'interaction_type'.")

        # Tiền xử lý dữ liệu
        # user_action['interaction_type'] = user_action['interaction_type'].map({'like': 1, 'comment': 0})
        reader = Reader(rating_scale=(0, 1))  # Mã hóa 'like' = 1 và 'comment' = 0
        data = Dataset.load_from_df(user_action[['userID', 'artID', 'interaction_type']], reader)

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        trainset, testset = train_test_split(data, test_size=test_size)

        # Khởi tạo mô hình SVD với các tham số
        svd = SVD(n_factors=n_factors, lr_all=lr_all, reg_all=reg_all)
        start_time = time.time()
        svd.fit(trainset)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")
        # Dự đoán trên tập kiểm tra
        predictions = svd.test(testset)
        rmse = accuracy.rmse(predictions)
        print(f"RMSE: {rmse:.4f}")
        return svd
# Hàm gợi ý các tác phẩm từ Collaborative Filtering (SVD)
    def get_collaborative_filtering_recommendations(self,user_id,user_action,svd, n=10):
        all_artworks = user_action['artID'].unique()
        already_rated = user_action[user_action['userID'] == user_id]['artID'].values
        artworks_to_predict = [artwork for artwork in all_artworks if artwork not in already_rated]

        predictions = [svd.predict(uid=user_id, iid=artwork_id) for artwork_id in artworks_to_predict]
        predictions.sort(key=lambda x: x.est, reverse=True)  # Sắp xếp theo điểm số từ cao đến thấp

        print(f"\nTop recommendations for User {user_id} based on Collaborative Filtering (SVD):")
        for pred in predictions[:n]:  # Lấy top 10 dự đoán
            print(f"Artwork: {pred.iid}, Predicted Rating: {pred.est}")

        return predictions
    

    def get_hybrid_recommendations_for_liked(self, user_id, artwork, user_action, svd, cosine_sim, n=10, weight_cf=0.4, weight_cb=0.6):
        if user_id not in user_action['userID'].values:
            print(f"No interaction data found for User {user_id}. Returning default recommendations.")
            return self.default_recommendation_for_user()  # Return default recommendations
        # Lấy các tác phẩm mà người dùng đã tương tác mạnh nhất qua "like"
        # 3.1: Content-Based Filtering - Lấy gợi ý từ Content-Based cho các tác phẩm đã like
        content_pred, content_similarities = self.get_content_based_recommendations_for_liked(user_id, user_action, artwork, cosine_sim, n)

        # 3.2: Collaborative Filtering - Lấy gợi ý từ Collaborative Filtering
        cf_predictions = self.get_collaborative_filtering_recommendations(user_id, user_action, svd, n)

        # 3.3: Kết hợp điểm số từ CF và CBF
        hybrid_predictions = []
        all_artworks = set(artwork['artID'].values)  # Tất cả các tác phẩm có trong cơ sở dữ liệu

        # Lấy điểm hybrid cho các tác phẩm đã được "like"
        for i, (artwork_id, content_score) in enumerate(zip(content_pred, content_similarities)):
            # Kiểm tra và chuyển loại dữ liệu cho artwork_id nếu là numpy.int64
            if isinstance(artwork_id, np.int64):
                artwork_id = int(artwork_id)

            # Tìm điểm số Collaborative Filtering cho từng artwork_id
            cf_score = next((pred.est for pred in cf_predictions if pred.iid == artwork_id), 0)

            # Tính điểm hybrid kết hợp giữa CF và CBF
            hybrid_score = (weight_cf * cf_score) + (weight_cb * content_score)

            hybrid_predictions.append((artwork_id, hybrid_score))

        # 3.4: Lấy tất cả các tác phẩm còn lại trong cơ sở dữ liệu mà không phải là những tác phẩm được "like"
        artworks_not_in_recommendations = [artwork_id for artwork_id in all_artworks if artwork_id not in content_pred]

        # 3.5: Lấy top n gợi ý từ Hybrid Recommendations
        hybrid_predictions.sort(key=lambda x: x[1], reverse=True)  # Sắp xếp theo hybrid score
        top_n_recommendations = hybrid_predictions[:n]

        # 3.6: Xây dựng danh sách tất cả các tác phẩm với các gợi ý ở đầu
        recommended_artworks = top_n_recommendations + [(artwork_id, 0) for artwork_id in artworks_not_in_recommendations]  # Các tác phẩm không có trong top n, nhưng xếp sau

        # 3.7: Trả về chỉ artworkID
        recommended_artworks_ids = [artwork_id for artwork_id, score in recommended_artworks]

        # Đảm bảo trả về một danh sách, kể cả khi chỉ có một artworkID
        return recommended_artworks_ids  # Trả về danh sách artworkID mà không có điểm số
    def default_recommendation_for_user(self):
        cache_key = "all_artworks"
        data = redis_client.get(cache_key)  # Lấy dữ liệu từ Redis
        if data:
            try:
                # If data is a JSON string, parse it into a Python object
                parsed_artworks = self.parse_artworks(data)  # Parse JSON string
                if parsed_artworks:
                    logger.info(f"Successfully parsed {len(parsed_artworks)} artworks from cache.")
                    return parsed_artworks  # Return parsed artworks
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding cached artwork data: {e}")
            except Exception as e:
                logger.error(f"Unexpected error when processing cache data: {e}")
        else:
            logger.info("No data found in cache.")

        return []  # Return an empty list if no data is found




