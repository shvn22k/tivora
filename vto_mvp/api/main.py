"""
Simple Virtual Try-On API
Just makes Replicate API calls for batch try-ons
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import replicate
import asyncio
import aiohttp
import os
import time
from pathlib import Path

# Config
# REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
REPLICATE_API_TOKEN = "lol"
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "10"))
OUTPUT_DIR = Path("outputs")
TEMP_DIR = Path("temp")

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# App
app = FastAPI(title="VTO API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Semaphore for concurrency control
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# Models
class Garment(BaseModel):
    garment_id: str
    garment_image_url: str
    garment_description: Optional[str] = "clothing item"
    category: str = "upper_body"  # upper_body, lower_body, dresses


class BatchRequest(BaseModel):
    user_id: str
    user_image_url: str
    garments: List[Garment]


class TryOnResult(BaseModel):
    garment_id: str
    success: bool
    result_url: Optional[str] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    user_id: str
    total: int
    successful: int
    failed: int
    results: List[TryOnResult]
    processing_time: float


# Helper functions
async def download_image(url: str, path: str):
    """Download image from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                with open(path, 'wb') as f:
                    f.write(await resp.read())
                return path
            raise Exception(f"Failed to download: {resp.status}")


def run_replicate_tryon(user_img: str, garment_img: str, description: str, category: str, output_path: str):
    """Call Replicate API (sync)"""
    with open(garment_img, "rb") as garm, open(user_img, "rb") as human:
        output = replicate.run(
            "cuuupid/idm-vton:0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            input={
                "garm_img": garm,
                "human_img": human,
                "garment_des": description,
                "category": category,
                "crop": False,
                "seed": 42,
                "steps": 30,
                "force_dc": False,
                "mask_only": False
            }
        )
    
    with open(output_path, "wb") as f:
        f.write(output.read())
    
    return output_path


async def process_single_tryon(user_id: str, user_image_url: str, garment: Garment):
    """Process one try-on with semaphore"""
    async with semaphore:
        try:
            # Download images
            user_path = TEMP_DIR / f"{user_id}_user.jpg"
            garment_path = TEMP_DIR / f"{garment.garment_id}_garment.jpg"
            
            await asyncio.gather(
                download_image(user_image_url, str(user_path)),
                download_image(garment.garment_image_url, str(garment_path))
            )
            
            # Generate result
            output_filename = f"{user_id}_{garment.garment_id}_{int(time.time())}.jpg"
            output_path = OUTPUT_DIR / output_filename
            
            # Run Replicate (in thread to not block)
            await asyncio.to_thread(
                run_replicate_tryon,
                str(user_path),
                str(garment_path),
                garment.garment_description,
                garment.category,
                str(output_path)
            )
            
            # Cleanup temp files
            user_path.unlink(missing_ok=True)
            garment_path.unlink(missing_ok=True)
            
            return TryOnResult(
                garment_id=garment.garment_id,
                success=True,
                result_url=f"/outputs/{output_filename}"
            )
            
        except Exception as e:
            return TryOnResult(
                garment_id=garment.garment_id,
                success=False,
                error=str(e)
            )


# Endpoints
@app.get("/")
def root():
    return {"status": "ok", "message": "VTO API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/v1/tryon/batch", response_model=BatchResponse)
async def batch_tryon(request: BatchRequest):
    """
    Batch try-on: process 50-60 garments concurrently
    """
    if not REPLICATE_API_TOKEN:
        raise HTTPException(500, "REPLICATE_API_TOKEN not set")
    
    start_time = time.time()
    
    # Process all garments concurrently
    tasks = [
        process_single_tryon(request.user_id, request.user_image_url, garment)
        for garment in request.garments
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Count successes
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    return BatchResponse(
        user_id=request.user_id,
        total=len(results),
        successful=successful,
        failed=failed,
        results=results,
        processing_time=time.time() - start_time
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
