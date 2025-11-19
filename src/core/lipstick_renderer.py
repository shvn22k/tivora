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

def apply_dynamic_lipstick(frame, lip_mask_float, skin_lab, light_type='neutral', coords=None, intensity=0.7):
    """
    Apply realistic lipstick with dynamic color selection and gloss effect.
    frame: BGR uint8
    lip_mask_float: 0..1 float mask (HxW)
    skin_lab: LAB mean (array 3)
    light_type: 'warm'|'cool'|'neutral'
    coords: landmark coordinates for gloss highlight
    returns new BGR frame
    """
    if lip_mask_float is None or lip_mask_float.max() < 0.01:
        return frame
    
    h, w = frame.shape[:2]
    
    # Determine lip color dynamically
    lip_rgb = determine_lip_color(skin_lab, light_type)
    lip_color = np.zeros_like(frame, dtype=np.uint8)
    lip_color[:] = lip_rgb[::-1]  # RGB → BGR
    
    # Blend in LAB space for natural shading
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lip_lab = cv2.cvtColor(lip_color, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask3 = np.repeat(lip_mask_float[:, :, None], 3, axis=2)
    
    blended_lab = (1 - mask3 * intensity) * frame_lab + (mask3 * intensity) * lip_lab
    blended_bgr = cv2.cvtColor(np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Add soft highlight for gloss effect
    gloss_mask = np.zeros((h, w), dtype=np.float32)
    if coords is not None:
        try:
            # Use lip center landmarks for gloss
            for idx in [13, 14]:
                if idx < len(coords):
                    cx, cy = coords[idx][:2]
                    cv2.circle(gloss_mask, (int(cx), int(cy - 5)), int(h * 0.02), 255, -1)
        except (IndexError, KeyError):
            pass
    
    if gloss_mask.max() > 0:
        gloss_mask = cv2.GaussianBlur(gloss_mask, (81, 81), 0) / 255.0
        gloss_layer = blended_bgr.astype(np.float32)
        gloss_layer[:, :, 0] += 8 * gloss_mask
        gloss_layer[:, :, 1] += 8 * gloss_mask
        gloss_layer[:, :, 2] += 8 * gloss_mask
        blended_bgr = np.clip(gloss_layer, 0, 255).astype(np.uint8)
    
    out = frame.copy()
    mask_bool = lip_mask_float > 0.01
    out[mask_bool] = blended_bgr[mask_bool]
    
    return out
