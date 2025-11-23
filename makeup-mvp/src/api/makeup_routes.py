# src/api/makeup_routes.py
"""
FastAPI routes for makeup application endpoints.
"""
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from typing import Optional
import io
import json
from src.services.makeup_service import makeup_service
from src.api.schemas import MakeupRequest, ColorPaletteResponse

router = APIRouter(prefix="/makeup", tags=["makeup"])


def image_to_bytes(image: np.ndarray, format: str = "JPEG") -> bytes:
    """Convert numpy image array to bytes."""
    is_success, buffer = cv2.imencode(f".{format.lower()}", image)
    if not is_success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to numpy image array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


@router.post("/apply", response_class=Response)
async def apply_makeup(
    file: UploadFile = File(..., description="Image file to process"),
    apply_lipstick: bool = True,
    apply_blush: bool = True,
    lipstick_r: Optional[int] = None,
    lipstick_g: Optional[int] = None,
    lipstick_b: Optional[int] = None,
    blush_r: Optional[int] = None,
    blush_g: Optional[int] = None,
    blush_b: Optional[int] = None,
    lipstick_intensity: float = 0.9,
    blush_intensity: float = 1.0,
):
    """
    Apply makeup to an uploaded image.
    
    Returns the processed image with makeup applied.
    """
    # Read image
    image_bytes = await file.read()
    image = bytes_to_image(image_bytes)
    
    # Parse color parameters
    lipstick_color = None
    if lipstick_r is not None and lipstick_g is not None and lipstick_b is not None:
        lipstick_color = (lipstick_r, lipstick_g, lipstick_b)
    
    blush_color = None
    if blush_r is not None and blush_g is not None and blush_b is not None:
        blush_color = (blush_r, blush_g, blush_b)
    
    # Process image
    result = makeup_service.process_image(
        image=image,
        lipstick_color=lipstick_color,
        blush_color=blush_color,
        apply_lipstick=apply_lipstick,
        apply_blush=apply_blush,
        lipstick_intensity=lipstick_intensity,
        blush_intensity=blush_intensity,
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    # Convert result to bytes
    output_bytes = image_to_bytes(result['image'])
    
    # Determine content type
    content_type = file.content_type or "image/jpeg"
    
    return Response(content=output_bytes, media_type=content_type)


@router.post("/apply-json", response_model=dict)
async def apply_makeup_json(
    file: UploadFile = File(..., description="Image file to process"),
    settings: Optional[str] = Form(None),
):
    """
    Apply makeup to an uploaded image with JSON request body.
    
    Returns the processed image as base64 and metadata.
    """
    import base64
    
    # Read image
    image_bytes = await file.read()
    image = bytes_to_image(image_bytes)
    
    # Parse request
    if settings:
        try:
            payload = json.loads(settings)
            request = MakeupRequest(**payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid settings payload: {exc}")
    else:
        request = MakeupRequest()
    
    # Process image
    result = makeup_service.process_image(
        image=image,
        lipstick_color=request.lipstick_color.to_tuple() if request.lipstick_color else None,
        blush_color=request.blush_color.to_tuple() if request.blush_color else None,
        apply_lipstick=request.apply_lipstick,
        apply_blush=request.apply_blush,
        lipstick_intensity=request.lipstick_intensity,
        blush_intensity=request.blush_intensity,
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    # Convert image to base64
    output_bytes = image_to_bytes(result['image'])
    image_base64 = base64.b64encode(output_bytes).decode('utf-8')
    
    return {
        'success': True,
        'image_base64': image_base64,
        'metadata': result['metadata']
    }


@router.post("/palettes", response_model=ColorPaletteResponse)
async def get_color_palettes(
    file: UploadFile = File(..., description="Image file to analyze")
):
    """
    Get color palettes for an image without applying makeup.
    
    Returns suggested color palettes based on detected skin tone.
    """
    # Read image
    image_bytes = await file.read()
    image = bytes_to_image(image_bytes)
    
    # Get palettes
    result = makeup_service.get_color_palettes(image)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    return ColorPaletteResponse(
        success=True,
        palettes=result['palettes'],
        skin_tone_hex=result['skin_tone_hex'],
        lighting_type=result['lighting_type'],
        brightness=result['brightness'],
    )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "makeup"}

