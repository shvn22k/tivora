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
    Apply realistic lipstick with smooth, uniform color application.
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
        base_intensity = 0.75  # Visible but natural
    else:
        lip_rgb = determine_lip_color(skin_lab, light_type)
        base_intensity = intensity * 0.8  # Visible but natural
    
    # Convert RGB to BGR
    lip_bgr = np.array([lip_rgb[2], lip_rgb[1], lip_rgb[0]], dtype=np.uint8)
    
    # Create smooth, uniform mask with soft edges
    # Use larger blur for smoother transitions
    smooth_mask = cv2.GaussianBlur(lip_mask_float, (25, 25), 0)
    smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
    
    # Convert frame to LAB for better color blending
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Convert lip color to LAB
    lip_bgr_single = np.uint8([[lip_bgr]])
    lip_lab_single = cv2.cvtColor(lip_bgr_single, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    
    # Create uniform lip color overlay in LAB space
    lip_lab = np.zeros((h, w, 3), dtype=np.float32)
    lip_lab[:, :, :] = lip_lab_single
    
    # Apply uniform intensity across the mask
    mask_3d = np.repeat(smooth_mask[:, :, None], 3, axis=2)
    intensity_map = mask_3d * base_intensity
    
    # Blend in LAB space for natural color mixing
    blended_lab = (1 - intensity_map) * frame_lab + intensity_map * lip_lab
    blended_bgr = cv2.cvtColor(np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Add subtle gloss highlight only in center (very subtle)
    if coords is not None:
        try:
            # Find lip center from landmarks
            lip_center_x = int(np.mean([coords[i][0] for i in [13, 14, 17, 18] if i < len(coords)]))
            lip_center_y = int(np.mean([coords[i][1] for i in [13, 14, 17, 18] if i < len(coords)]))
            
            gloss_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(gloss_mask, (lip_center_x, lip_center_y - int(h * 0.01)), 
                      int(h * 0.02), 1.0, -1)
            gloss_mask = cv2.GaussianBlur(gloss_mask, (31, 31), 0)
            gloss_mask = np.clip(gloss_mask * 0.3, 0.0, 1.0)  # Very subtle
            
            # Add slight brightness increase for gloss
            gloss_3d = np.repeat(gloss_mask[:, :, None], 3, axis=2)
            blended_bgr = blended_bgr.astype(np.float32) + gloss_3d * np.array([8, 6, 5])
            blended_bgr = np.clip(blended_bgr, 0, 255).astype(np.uint8)
        except (IndexError, KeyError, ValueError):
            pass
    
    # Final blend with smooth mask edges
    # Use the original smooth mask for blending
    final_mask_3d = np.repeat(smooth_mask[:, :, None], 3, axis=2)
    
    # Blend: preserve some original texture but apply lipstick uniformly
    out = frame.astype(np.float32) * (1 - final_mask_3d * 0.8) + blended_bgr.astype(np.float32) * (final_mask_3d * 0.8)
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out
