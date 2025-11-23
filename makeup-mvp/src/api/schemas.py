# src/api/schemas.py
"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class ColorRGB(BaseModel):
    """RGB color model."""
    r: int = Field(..., ge=0, le=255, description="Red component (0-255)")
    g: int = Field(..., ge=0, le=255, description="Green component (0-255)")
    b: int = Field(..., ge=0, le=255, description="Blue component (0-255)")
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)


class MakeupRequest(BaseModel):
    """Request model for makeup application."""
    apply_lipstick: bool = Field(True, description="Whether to apply lipstick")
    apply_blush: bool = Field(True, description="Whether to apply blush")
    lipstick_color: Optional[ColorRGB] = Field(None, description="Custom lipstick color (RGB)")
    blush_color: Optional[ColorRGB] = Field(None, description="Custom blush color (RGB)")
    lipstick_intensity: float = Field(0.9, ge=0.0, le=1.0, description="Lipstick intensity (0.0-1.0)")
    blush_intensity: float = Field(1.0, ge=0.0, le=1.0, description="Blush intensity (0.0-1.0)")


class ColorPaletteResponse(BaseModel):
    """Response model for color palette."""
    success: bool
    palettes: Optional[dict] = None
    skin_tone_hex: Optional[str] = None
    lighting_type: Optional[str] = None
    brightness: Optional[float] = None
    message: Optional[str] = None


class MakeupResponse(BaseModel):
    """Response model for makeup application."""
    success: bool
    message: Optional[str] = None
    metadata: Optional[dict] = None

