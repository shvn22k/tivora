"""
Minimal VTO API - Single Try-On Endpoint
Port: 8003
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import replicate
import requests
import os
from pathlib import Path
import time

# Setup
app = FastAPI(title="VTO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output and temp directories
OUTPUT_DIR = Path("outputs/vto")
TEMP_DIR = Path("temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Hardcoded API key for demo
REPLICATE_API_TOKEN = "r8_0a5uWF6Vu6VZcbIz7AVw4XqltIfJW5v4R9ocu"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


# Models
class TryOnRequest(BaseModel):
    person_image_url: str
    garment_image_url: str
    garment_description: Optional[str] = "clothing item"
    category: str = "upper_body"  # upper_body, lower_body, dresses


class TryOnResponse(BaseModel):
    success: bool
    result_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: float


# Endpoints
@app.get("/")
def health_check():
    return {"service": "VTO API", "port": 8003, "status": "running"}


@app.post("/tryon", response_model=TryOnResponse)
def try_on(request: TryOnRequest):
    """
    Single try-on endpoint
    """
    start_time = time.time()
    
    try:
        # Call Replicate directly with URLs
        output = replicate.run(
            "cuuupid/idm-vton:0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            input={
                "garm_img": request.garment_image_url,
                "human_img": request.person_image_url,
                "garment_des": request.garment_description,
                "category": request.category,
                "crop": False,
                "seed": 42,
                "steps": 30,
                "force_dc": False,
                "mask_only": False
            }
        )
        
        # Download result
        result_url = str(output)  # Replicate returns a URL
        result_bytes = requests.get(result_url).content
        
        # Save result locally
        output_filename = f"tryon_{int(time.time())}.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        with open(output_path, "wb") as f:
            f.write(result_bytes)
        
        return TryOnResponse(
            success=True,
            result_url=f"/outputs/vto/{output_filename}",
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        return TryOnResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
