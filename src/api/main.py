# src/api/main.py
"""
FastAPI main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.makeup_routes import router as makeup_router
from src.api.video_routes import router as video_router

app = FastAPI(
    title="TIVORA AR Makeup API",
    description="API for applying AR makeup to images and real-time video streams",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(makeup_router)
app.include_router(video_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TIVORA AR Makeup API",
        "version": "1.0.0",
        "docs": "/docs"
    }

