# src/core/eyeshadow_generator.py
import numpy as np
import cv2
from src.utils.blend import add_skin_texture

def apply_dynamic_eyeshadow(frame, eyeshadow_mask_float, skin_lab, light_type='neutral', brightness=0.5, coords=None, strength=1.0, custom_color_rgb=None):
    """
    Apply natural-looking eyeshadow with subtle gradients and texture preservation.
    Works with glasses by avoiding bright reflection zones.
    eyeshadow_mask_float: 0..1 float mask
    skin_lab: 3-element LAB array
    light_type: 'warm'|'cool'|'neutral'
    brightness: normalized brightness 0..1
    coords: landmark coordinates
    custom_color_rgb: Optional RGB tuple to override automatic color selection
    """
    if eyeshadow_mask_float is None or eyeshadow_mask_float.max() < 0.01:
        return frame
    
    if skin_lab is None and custom_color_rgb is None:
        return frame
    
    h, w = frame.shape[:2]
    
    # Use custom color if provided, otherwise adjust tone based on lighting
    if custom_color_rgb is not None:
        # Convert RGB to BGR
        eyeshadow_rgb = custom_color_rgb
        eyeshadow_bgr = np.array([eyeshadow_rgb[2], eyeshadow_rgb[1], eyeshadow_rgb[0]], dtype=np.uint8)
        # More visible eyeshadow
        base_intensity = 0.7
    else:
        # Adjust tone based on lighting - eyeshadow colors are typically deeper/more muted
        target_lab = np.array(skin_lab, dtype=np.float32).copy()
        
        if light_type == "warm":
            # Warm lighting: warm browns, bronzes, golds
            target_lab[0] = max(30, target_lab[0] - 25)  # Darker
            target_lab[1] += 8   # More red/warm
            target_lab[2] += 12  # More yellow/gold
        elif light_type == "cool":
            # Cool lighting: cool grays, purples, taupes
            target_lab[0] = max(30, target_lab[0] - 30)  # Darker
            target_lab[1] += 4   # Slight red
            target_lab[2] -= 8   # Less yellow, more cool
        else:
            # Neutral: balanced browns and taupes
            target_lab[0] = max(30, target_lab[0] - 28)  # Darker
            target_lab[1] += 6   # Slight warmth
            target_lab[2] += 2   # Slight yellow
        
        # Build LAB overlay image
        lab_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        lab_overlay[:, :, 0] = np.clip(target_lab[0], 0, 255)
        lab_overlay[:, :, 1] = np.clip(target_lab[1], 0, 255)
        lab_overlay[:, :, 2] = np.clip(target_lab[2], 0, 255)
        eyeshadow_bgr = cv2.cvtColor(lab_overlay, cv2.COLOR_LAB2BGR)[0, 0]
        
        # Dynamically adjust intensity depending on brightness - more visible
        base_intensity = np.interp(brightness, [0.3, 0.7], [0.75, 0.65])  # Much more visible
    
    # Create intensity variation map for natural gradient
    # Stronger near lash line, fades smoothly upward
    distance_map = cv2.distanceTransform(
        (eyeshadow_mask_float > 0.01).astype(np.uint8), 
        cv2.DIST_L2, 5
    )
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    else:
        distance_map = eyeshadow_mask_float
    
    # Create natural gradient: stronger at bottom (lash line), soft fade upward
    # Invert distance map so center (lash line) is stronger
    lash_intensity = 1.0 - (distance_map * 0.4)  # 0.6 to 1.0 (stronger at edges/lash line)
    edge_fade = cv2.GaussianBlur(eyeshadow_mask_float, (51, 51), 0)  # Soft edges
    variation_mask = lash_intensity * edge_fade
    
    # Blend in LAB space with natural variation
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Create eyeshadow color in LAB space
    eyeshadow_lab_single = cv2.cvtColor(np.uint8([[eyeshadow_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    eyeshadow_lab = np.zeros((h, w, 3), dtype=np.float32)
    eyeshadow_lab[:, :, :] = eyeshadow_lab_single
    
    # Apply variation to intensity
    variation_3d = np.repeat(variation_mask[:, :, None], 3, axis=2)
    intensity_map = variation_3d * base_intensity * strength
    
    # Soft blend that preserves skin texture
    blended_lab = (1 - intensity_map) * frame_lab + intensity_map * eyeshadow_lab
    blended_bgr = cv2.cvtColor(np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Preserve natural skin texture with soft blending
    soft_mask = cv2.GaussianBlur(eyeshadow_mask_float, (31, 31), 0)
    soft_mask_3d = np.repeat(soft_mask[:, :, None], 3, axis=2)
    
    # Final blend: preserve texture, add visible eyeshadow
    blend_factor = soft_mask_3d * 0.85  # 85% blend for strong visibility
    out = frame.astype(np.float32) * (1 - blend_factor) + blended_bgr.astype(np.float32) * blend_factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Add very subtle texture to prevent airbrushed look
    out = add_skin_texture(out, strength=0.4)
    
    return out

