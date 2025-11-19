# src/core/blush_generator.py
import numpy as np
import cv2
from src.utils.blend import overlay_blend, add_skin_texture

def apply_dynamic_blush(frame, cheek_mask_float, skin_lab, light_type='neutral', brightness=0.5, coords=None, strength=1.0):
    """
    Apply dynamic, lighting-aware blush with overlay blend and texture.
    cheek_mask_float: 0..1 float mask
    skin_lab: 3-element LAB array
    light_type: 'warm'|'cool'|'neutral'
    brightness: normalized brightness 0..1
    coords: landmark coordinates for optional nose glow
    """
    if cheek_mask_float is None or cheek_mask_float.max() < 0.01:
        return frame
    
    if skin_lab is None:
        return frame
    
    # Adjust tone based on lighting
    target_lab = np.array(skin_lab, dtype=np.float32).copy()
    
    if light_type == "warm":
        # Warmer tone = soft coral-pink
        target_lab[1] += 9    # add red
        target_lab[2] -= 3    # reduce yellow
    elif light_type == "cool":
        # Cooler tone = mauve / rose
        target_lab[1] += 7
        target_lab[2] -= 7
    else:
        # Neutral = natural pink-beige
        target_lab[1] += 8
        target_lab[2] -= 5
    
    # Dynamically adjust intensity depending on brightness
    # (dim light → stronger tint)
    intensity_factor = np.interp(brightness, [0.3, 0.7], [1.4, 0.8])
    
    # Build LAB overlay image
    h, w = frame.shape[:2]
    lab_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    lab_overlay[:, :, 0] = np.clip(target_lab[0], 0, 255)
    lab_overlay[:, :, 1] = np.clip(target_lab[1], 0, 255)
    lab_overlay[:, :, 2] = np.clip(target_lab[2], 0, 255)
    blush_bgr_img = cv2.cvtColor(lab_overlay, cv2.COLOR_LAB2BGR)
    
    # Use overlay blend mode
    mask3_b = np.repeat(cheek_mask_float[:, :, None], 3, axis=2)
    blurred_mask = cv2.GaussianBlur(mask3_b, (81, 81), 0)
    
    # Blend over current frame
    overlay_img = blush_bgr_img
    blended = overlay_blend(frame, overlay_img)
    blended = frame.astype(np.float32) * (1 - blurred_mask) + blended.astype(np.float32) * blurred_mask * intensity_factor
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Subtle skin texture reintroduction
    out = add_skin_texture(blended, strength=0.8)
    
    # Optional: add soft nose glow only in warm lighting
    if light_type == "warm" and coords is not None:
        try:
            nose_x, nose_y = int(coords[1][0]), int(coords[1][1])
            h, w = frame.shape[:2]
            cv2.circle(out, (nose_x, nose_y + int(h * 0.01)),
                       int(w * 0.035), (15, 25, 60), -1)
            out = cv2.GaussianBlur(out, (15, 15), 0)
        except (IndexError, KeyError):
            pass
    
    return out
