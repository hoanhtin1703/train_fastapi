from fastapi import FastAPI
from router import artwork,user_action,recommend_user

app = FastAPI()

# Đăng ký các routers
app.include_router(artwork.router, prefix="/artworks", tags=["Artworks"])


app.include_router(user_action.router, prefix="/user_action", tags=["Users"])
app.include_router(recommend_user.router, prefix="/recommendation", tags=["Recommendation"])
