# src/services/makeup_service.py
"""
Makeup service for processing images with AR makeup application.
Extracted from makeup_live.py for API usage.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict, Any
from src.core.masks import (
    create_dynamic_blush_mask,
    to_int_coords,
    polygon_mask_from_points,
)
from src.core.skin_tone import (
    dynamic_skin_tone_tracker,
    get_skin_color_bgr,
)
from src.core.lighting import detect_lighting
from src.core.lipstick_renderer import apply_dynamic_lipstick
from src.core.blush_generator import apply_dynamic_blush
from src.core.color_palette import (
    generate_lipstick_palette,
    generate_blush_palette,
)

# MediaPipe face mesh (static mode for single images)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Full lip landmark indices from MediaPipe
UPPER_LIP_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
LOWER_LIP_IDX = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
]


class MakeupService:
    """Service for applying AR makeup to images."""
    
    def __init__(self):
        self.face_mesh = face_mesh
        self.prev_skin_lab = None
        self.prev_skin_hex = None
        self.prev_lipstick_palette = None
        self.prev_blush_palette = None
        self.prev_metadata = None
        self.stability_threshold = 8.0  # Higher threshold - only update on significant changes
        self.frame_count = 0
        self.skin_tone_locked = False  # Lock skin tone once stable
        self.stable_frames = 0  # Count consecutive stable frames
    
    def process_image(
        self,
        image: np.ndarray,
        lipstick_color: Optional[Tuple[int, int, int]] = None,
        blush_color: Optional[Tuple[int, int, int]] = None,
        apply_lipstick: bool = True,
        apply_blush: bool = True,
        lipstick_intensity: float = 0.9,
        blush_intensity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process an image and apply makeup.
        
        Args:
            image: BGR image as numpy array
            lipstick_color: Optional RGB tuple for lipstick color
            blush_color: Optional RGB tuple for blush color
            apply_lipstick: Whether to apply lipstick
            apply_blush: Whether to apply blush
            lipstick_intensity: Lipstick intensity (0.0-1.0)
            blush_intensity: Blush intensity (0.0-1.0)
        
        Returns:
            Dictionary with processed image and metadata
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out = image.copy()
        
        # Detect face landmarks
        results = self.face_mesh.process(rgb)
        coords = None
        face_detected = False
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = to_int_coords(lm, image.shape)
            face_detected = True
        
        if not face_detected:
            if self.prev_metadata is not None:
                fallback_metadata = dict(self.prev_metadata)
                fallback_metadata['stability_status'] = 'cached_face_missing'
                return {
                    'success': True,
                    'image': image,
                    'metadata': fallback_metadata
                }
            return {
                'success': False,
                'message': 'No face detected in image',
                'image': image,
                'metadata': {}
            }
        
        # Get skin tone - fast initial detection, then lock it
        self.frame_count += 1
        
        # Fast initial detection - use direct BGR sampling for first frame
        if self.prev_skin_lab is None:
            # First frame - get skin tone directly, no smoothing
            skin_bgr, skin_hex = get_skin_color_bgr(image, coords)
            if skin_bgr is not None:
                # Convert BGR to LAB directly
                bgr_patch = np.uint8([[skin_bgr]])
                skin_lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                self.prev_skin_lab = skin_lab
                self.prev_skin_hex = skin_hex
                self.stable_frames = 1
            else:
                skin_lab = None
                skin_hex = "#000000"
        else:
            # Subsequent frames - use temporal smoothing only if not locked
            if self.skin_tone_locked:
                # Skin tone is locked - use cached values
                skin_lab = self.prev_skin_lab
                skin_bgr, skin_hex = get_skin_color_bgr(image, coords)
                # Use cached hex for consistency
                skin_hex = self.prev_skin_hex
            else:
                # Still detecting - use fast convergence
                prev_lab = self.prev_skin_lab
                # Fast alpha for quick convergence (0.7 = 70% new, 30% old)
                alpha = 0.7 if self.frame_count < 5 else 0.85
                skin_lab = dynamic_skin_tone_tracker(image, coords, prev_lab=prev_lab, alpha=alpha, locked=False)
                skin_bgr, skin_hex = get_skin_color_bgr(image, coords)
                
                # Check stability - lock after 3-5 stable frames
                if skin_lab is not None:
                    delta = float(np.linalg.norm(skin_lab - prev_lab))
                    if delta < 3.0:  # Very stable
                        self.stable_frames += 1
                        if self.stable_frames >= 3:  # Lock after 3 stable frames
                            self.skin_tone_locked = True
                            self.prev_skin_lab = skin_lab
                            self.prev_skin_hex = skin_hex
                    else:
                        # Not stable yet - reset counter
                        self.stable_frames = 0
                        self.prev_skin_lab = skin_lab
                        self.prev_skin_hex = skin_hex
                else:
                    skin_lab = self.prev_skin_lab
                    skin_hex = self.prev_skin_hex
        
        # Detect lighting
        light_type, brightness = detect_lighting(image)
        
        # Generate color palettes (only once when skin tone is locked or first detected)
        lipstick_palette = None
        blush_palette = None
        stability_status = 'fresh'

        if skin_lab is not None:
            # Check if we need to regenerate palettes (only on significant change or first time)
            if self.prev_skin_lab is not None and not self.skin_tone_locked:
                delta = float(np.linalg.norm(skin_lab - self.prev_skin_lab))
                if delta < self.stability_threshold:
                    # Use cached palettes
                    lipstick_palette = self.prev_lipstick_palette
                    blush_palette = self.prev_blush_palette
                    stability_status = 'cached'
                else:
                    # Significant change - regenerate
                    stability_status = 'updated'
            elif self.skin_tone_locked:
                # Locked - always use cached
                lipstick_palette = self.prev_lipstick_palette
                blush_palette = self.prev_blush_palette
                stability_status = 'locked'

            # Generate palettes if not cached
            if lipstick_palette is None:
                lipstick_palette = generate_lipstick_palette(skin_lab, light_type)
                self.prev_lipstick_palette = lipstick_palette
            if blush_palette is None:
                blush_palette = generate_blush_palette(skin_lab, light_type)
                self.prev_blush_palette = blush_palette
        else:
            # Fallback to previous palettes when LAB extraction fails
            if self.prev_skin_hex:
                skin_hex = self.prev_skin_hex
            lipstick_palette = self.prev_lipstick_palette
            blush_palette = self.prev_blush_palette
            if lipstick_palette or blush_palette:
                stability_status = 'cached_no_skin_lab'
        # Auto-select default shades if custom colors not provided
        selected_lipstick_color = lipstick_color
        if selected_lipstick_color is None and lipstick_palette:
            selected_lipstick_color = tuple(lipstick_palette[0])

        selected_blush_color = blush_color
        if selected_blush_color is None and blush_palette:
            selected_blush_color = tuple(blush_palette[0])
        
        # Apply lipstick
        if apply_lipstick:
            lip_points = [coords[i][:2] for i in UPPER_LIP_IDX + LOWER_LIP_IDX[::-1] if i < len(coords)]
            if len(lip_points) >= 6:
                lip_mask_raw = polygon_mask_from_points(image.shape, lip_points)
                lip_mask = cv2.GaussianBlur(lip_mask_raw, (65, 65), 0).astype(np.float32) / 255.0
                
                out = apply_dynamic_lipstick(
                    out, lip_mask, skin_lab,
                    light_type=light_type, coords=coords, intensity=lipstick_intensity,
                    custom_color_rgb=selected_lipstick_color
                )
        
        # Apply blush
        if apply_blush:
            cheek_mask = create_dynamic_blush_mask(image, coords)
            if cheek_mask is not None:
                out = apply_dynamic_blush(
                    out, cheek_mask, skin_lab,
                    light_type=light_type, brightness=brightness,
                    coords=coords, strength=blush_intensity,
                    custom_color_rgb=selected_blush_color
                )
        
        # Prepare metadata
        metadata = {
            'face_detected': face_detected,
            'skin_tone_hex': skin_hex,
            'lighting_type': light_type,
            'brightness': float(brightness),
            'lipstick_palette': lipstick_palette,
            'blush_palette': blush_palette,
            'lipstick_color_used': selected_lipstick_color,
            'blush_color_used': selected_blush_color,
            'stability_status': stability_status,
        }

        if skin_lab is not None:
            self.prev_skin_lab = skin_lab.copy()
            self.prev_skin_hex = skin_hex
            self.prev_lipstick_palette = lipstick_palette
            self.prev_blush_palette = blush_palette
            self.prev_metadata = metadata.copy()
        
        return {
            'success': True,
            'image': out,
            'metadata': metadata
        }
    
    def get_color_palettes(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get color palettes for an image without applying makeup.
        
        Args:
            image: BGR image as numpy array
        
        Returns:
            Dictionary with color palettes and skin tone info
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return {
                'success': False,
                'message': 'No face detected in image',
                'palettes': {}
            }
        
        coords = to_int_coords(results.multi_face_landmarks[0].landmark, image.shape)
        skin_lab = dynamic_skin_tone_tracker(image, coords, prev_lab=None, alpha=0.95, locked=False)
        skin_bgr, skin_hex = get_skin_color_bgr(image, coords)
        light_type, brightness = detect_lighting(image)
        
        lipstick_palette = generate_lipstick_palette(skin_lab, light_type) if skin_lab is not None else None
        blush_palette = generate_blush_palette(skin_lab, light_type) if skin_lab is not None else None
        return {
            'success': True,
            'palettes': {
                'lipstick': lipstick_palette,
                'blush': blush_palette,
            },
            'skin_tone_hex': skin_hex,
            'lighting_type': light_type,
            'brightness': float(brightness),
        }


# Global service instance
makeup_service = MakeupService()

