from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import replicate
import requests
import os
from pathlib import Path
import time
import shutil

app = FastAPI(title="VTO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs/vto")
UPLOADS_DIR = Path("uploads/persons")
GARMENTS_DIR = Path("uploads/garments")
TEMP_DIR = Path("temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GARMENTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

REPLICATE_API_TOKEN = "r8_0a5uWF6Vu6VZcbIz7AVw4XqltIfJW5v4R9ocu"
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
class TryOnRequest(BaseModel):
    person_image_url: str
    garment_image_url: str
    garment_description: Optional[str] = "clothing item"
    category: str = "upper_body"

class TryOnResponse(BaseModel):
    success: bool
    result_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: float
@app.get("/")
def health_check():
    return {"service": "VTO API", "port": 8003, "status": "running"}


@app.post("/upload-person")
async def upload_person_image(file: UploadFile = File(...)):
    try:
        filename = f"person_{int(time.time())}_{file.filename}"
        file_path = UPLOADS_DIR / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        image_url = f"http://localhost:8003/uploads/persons/{filename}"
        
        return {
            "success": True,
            "image_url": image_url,
            "filename": filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tryon", response_model=TryOnResponse)
def try_on(request: TryOnRequest):
    start_time = time.time()
    garment_img_path = None
    person_img_path = None
    
    try:
        if "localhost" in request.garment_image_url:
            garment_img_path = GARMENTS_DIR / f"garment_{int(time.time())}.jpg"
            resp = requests.get(request.garment_image_url)
            with open(garment_img_path, "wb") as f:
                f.write(resp.content)
        
        if "localhost" in request.person_image_url:
            person_img_path = UPLOADS_DIR / f"person_{int(time.time())}.jpg"
            resp = requests.get(request.person_image_url)
            with open(person_img_path, "wb") as f:
                f.write(resp.content)
        
        with open(garment_img_path if garment_img_path else request.garment_image_url, "rb") as garm:
            with open(person_img_path if person_img_path else request.person_image_url, "rb") as person:
                output = replicate.run(
                    "cuuupid/idm-vton:0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
                    input={
                        "garm_img": garm,
                        "human_img": person,
                        "garment_des": request.garment_description,
                        "category": request.category,
                        "crop": False,
                        "seed": 42,
                        "steps": 30,
                        "force_dc": False,
                        "mask_only": False
                    }
                )
        
        result_url = str(output)
        result_bytes = requests.get(result_url).content
        
        output_filename = f"tryon_{int(time.time())}.jpg"
        output_path = OUTPUT_DIR / output_filename
        
        with open(output_path, "wb") as f:
            f.write(result_bytes)
        
        if garment_img_path and garment_img_path.exists():
            garment_img_path.unlink()
        if person_img_path and person_img_path.exists():
            person_img_path.unlink()
        
        return TryOnResponse(
            success=True,
            result_url=f"/outputs/vto/{output_filename}",
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        try:
            if garment_img_path and garment_img_path.exists():
                garment_img_path.unlink()
            if person_img_path and person_img_path.exists():
                person_img_path.unlink()
        except:
            pass
        
        return TryOnResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
