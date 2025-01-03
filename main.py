from fastapi import FastAPI
from router import  recommend_user

app = FastAPI()


app.include_router(recommend_user.router, prefix="/recommendation", tags=["Recommendation"])
                      