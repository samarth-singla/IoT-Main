from fastapi import FastAPI, HTTPException
from routes.route import router
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS middleware configuration
origins = [
    "http://localhost:3000",
    # Add your frontend URL when deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main router
app.include_router(router)

# Add endpoint to get ML predictions
@app.get("/ml-prediction")
async def get_ml_prediction():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:8001/predict")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="ML service error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
