# src/core/lipstick_renderer.py
import numpy as np
import cv2

def lab_to_bgr(lab_triplet):
    lab = np.uint8([[lab_triplet]])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0,0]
    return tuple(int(x) for x in bgr)

def determine_lip_color(skin_lab, light_type):
    """
    Determine lip color based on skin tone and lighting.
    Returns RGB tuple.
    """
    if skin_lab is None:
        return (150, 60, 80)  # default warm nude
    
    skin_L, skin_A, skin_B = skin_lab
    
    # Base tone selection based on skin lightness
    if skin_L > 170:
        # Very fair skin → soft rose/coral
        lip_rgb = (190, 90, 120)
    elif skin_L > 130:
        # Medium skin → warm nude/peach
        lip_rgb = (150, 60, 80)
    elif skin_L > 100:
        # Tan skin → mauve/berry
        lip_rgb = (120, 40, 90)
    else:
        # Deep skin → bold wine tone
        lip_rgb = (100, 20, 60)
    
    # Adjust tone slightly by lighting
    if light_type == "warm":
        lip_rgb = (min(lip_rgb[0]+15, 255), lip_rgb[1], max(lip_rgb[2]-10, 0))
    elif light_type == "cool":
        lip_rgb = (max(lip_rgb[0]-10, 0), lip_rgb[1], min(lip_rgb[2]+15, 255))
    
    return lip_rgb

def apply_dynamic_lipstick(frame, lip_mask_float, skin_lab, light_type='neutral', coords=None, intensity=0.7, custom_color_rgb=None):
    """
    Apply realistic lipstick with natural variation, depth, and gloss effect.
    frame: BGR uint8
    lip_mask_float: 0..1 float mask (HxW)
    skin_lab: LAB mean (array 3)
    light_type: 'warm'|'cool'|'neutral'
    coords: landmark coordinates for gloss highlight
    custom_color_rgb: Optional RGB tuple to override automatic color selection
    returns new BGR frame
    """
    if lip_mask_float is None or lip_mask_float.max() < 0.01:
        return frame
    
    h, w = frame.shape[:2]
    
    # Use custom color if provided, otherwise determine dynamically
    if custom_color_rgb is not None:
        lip_rgb = custom_color_rgb
        base_intensity = 0.65  # Visible but natural
    else:
        lip_rgb = determine_lip_color(skin_lab, light_type)
        base_intensity = intensity * 0.75  # Visible but natural
    
    # Create base lip color
    lip_color = np.zeros_like(frame, dtype=np.uint8)
    lip_color[:] = lip_rgb[::-1]  # RGB → BGR
    
    # Create intensity variation map for natural depth
    # Stronger at center, softer at edges
    distance_map = cv2.distanceTransform(
        (lip_mask_float > 0.01).astype(np.uint8), 
        cv2.DIST_L2, 5
    )
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    else:
        distance_map = lip_mask_float
    
    # Create gradient: stronger in center, fades at edges
    center_intensity = distance_map * 0.8 + 0.2  # 0.2 to 1.0
    edge_fade = cv2.GaussianBlur(lip_mask_float, (31, 31), 0)
    variation_mask = center_intensity * edge_fade
    
    # Blend in LAB space with variation
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lip_lab = cv2.cvtColor(lip_color, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Create variation in color intensity (darker at edges, lighter in center)
    variation_3d = np.repeat(variation_mask[:, :, None], 3, axis=2)
    intensity_map = variation_3d * base_intensity
    
    # Soft blend that preserves natural lip texture
    blended_lab = (1 - intensity_map) * frame_lab + intensity_map * lip_lab
    
    # Add subtle color variation (slightly darker at outer edges)
    edge_mask = 1.0 - (distance_map * 0.3)  # Darker at edges
    edge_mask_3d = np.repeat(edge_mask[:, :, None], 3, axis=2)
    blended_lab[:, :, 0] *= (0.95 + edge_mask_3d[:, :, 0] * 0.1)  # Slight darkening at edges
    
    blended_bgr = cv2.cvtColor(np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Add natural gloss highlight (stronger in center)
    gloss_mask = np.zeros((h, w), dtype=np.float32)
    if coords is not None:
        try:
            # Use lip center landmarks for gloss
            for idx in [13, 14]:
                if idx < len(coords):
                    cx, cy = coords[idx][:2]
                    cv2.circle(gloss_mask, (int(cx), int(cy - 5)), int(h * 0.025), 255, -1)
        except (IndexError, KeyError):
            pass
    
    if gloss_mask.max() > 0:
        gloss_mask = cv2.GaussianBlur(gloss_mask, (61, 61), 0) / 255.0
        # Multiply with distance map so gloss is stronger in center
        gloss_mask = gloss_mask * distance_map * 0.6
        
        # Add subtle brightness and color shift for gloss
        gloss_layer = blended_bgr.astype(np.float32)
        gloss_3d = np.repeat(gloss_mask[:, :, None], 3, axis=2)
        gloss_layer += gloss_3d * np.array([15, 12, 10])  # Subtle brightness increase
        blended_bgr = np.clip(gloss_layer, 0, 255).astype(np.uint8)
    
    # Preserve natural texture by blending with original
    # Use softer mask edges for seamless blend
    soft_mask = cv2.GaussianBlur(lip_mask_float, (15, 15), 0)
    soft_mask_3d = np.repeat(soft_mask[:, :, None], 3, axis=2)
    
    # Final blend: preserve more original texture but make it visible
    out = frame.astype(np.float32) * (1 - soft_mask_3d * 0.85) + blended_bgr.astype(np.float32) * (soft_mask_3d * 0.85)
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out
