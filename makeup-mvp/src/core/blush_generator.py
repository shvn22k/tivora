# src/core/blush_generator.py
import numpy as np
import cv2
from src.utils.blend import overlay_blend, add_skin_texture

def apply_dynamic_blush(frame, cheek_mask_float, skin_lab, light_type='neutral', brightness=0.5, coords=None, strength=1.0, custom_color_rgb=None):
    """
    Apply natural-looking blush with subtle gradients and texture preservation.
    cheek_mask_float: 0..1 float mask
    skin_lab: 3-element LAB array
    light_type: 'warm'|'cool'|'neutral'
    brightness: normalized brightness 0..1
    coords: landmark coordinates for optional nose glow
    custom_color_rgb: Optional RGB tuple to override automatic color selection
    """
    if cheek_mask_float is None or cheek_mask_float.max() < 0.01:
        return frame
    
    if skin_lab is None and custom_color_rgb is None:
        return frame
    
    h, w = frame.shape[:2]
    
    # Use custom color if provided, otherwise adjust tone based on lighting
    if custom_color_rgb is not None:
        # Convert RGB to BGR
        blush_rgb = custom_color_rgb
        blush_bgr = np.array([blush_rgb[2], blush_rgb[1], blush_rgb[0]], dtype=np.uint8)
        base_intensity = 0.65  # Clearly visible
    else:
        # Adjust tone based on lighting - more vibrant pink tint
        target_lab = np.array(skin_lab, dtype=np.float32).copy()
        
        if light_type == "warm":
            # Warmer tone = soft coral-pink
            target_lab[1] += 18    # more visible red tint
            target_lab[2] -= 2    # slight yellow reduction
        elif light_type == "cool":
            # Cooler tone = mauve / rose
            target_lab[1] += 17    # more visible red
            target_lab[2] -= 6    # cool shift
        else:
            # Neutral = natural pink-beige
            target_lab[1] += 17    # more visible red
            target_lab[2] -= 3    # slight cool shift
        
        # Build LAB overlay image
        lab_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        lab_overlay[:, :, 0] = np.clip(target_lab[0], 0, 255)
        lab_overlay[:, :, 1] = np.clip(target_lab[1], 0, 255)
        lab_overlay[:, :, 2] = np.clip(target_lab[2], 0, 255)
        blush_bgr = cv2.cvtColor(lab_overlay, cv2.COLOR_LAB2BGR)[0, 0]
        
        # Dynamically adjust intensity depending on brightness - clearly visible
        base_intensity = np.interp(brightness, [0.3, 0.7], [0.6, 0.5])
    
    # Create intensity variation map for natural gradient
    # Stronger in center, fades smoothly at edges
    distance_map = cv2.distanceTransform(
        (cheek_mask_float > 0.01).astype(np.uint8), 
        cv2.DIST_L2, 5
    )
    if distance_map.max() > 0:
        distance_map = distance_map / distance_map.max()
    else:
        distance_map = cheek_mask_float
    
    # Create natural gradient: stronger in center, soft fade at edges
    # More visible gradient for clearly noticeable blush
    center_intensity = distance_map * 0.6 + 0.4  # 0.4 to 1.0 for visible depth
    edge_fade = cv2.GaussianBlur(cheek_mask_float, (101, 101), 0)  # Soft edges
    variation_mask = np.clip(center_intensity * edge_fade, 0.0, 0.95)  # Cap at 0.95 for visibility
    
    # Convert to LAB for natural color blending
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Create blush color in LAB space
    blush_lab_single = cv2.cvtColor(np.uint8([[blush_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    
    # Apply strong visible blending - like real blush
    # Blend color channels while slightly enhancing brightness for natural glow
    variation_3d = np.repeat(variation_mask[:, :, None], 3, axis=2)
    intensity_map = np.clip(variation_3d * base_intensity * strength, 0.0, 0.85)  # Very visible max 85%
    
    # Blend color channels with slight brightness enhancement for natural glow
    blended_lab = frame_lab.copy()
    # Slightly enhance brightness in blush areas for natural glow
    brightness_boost = intensity_map[:, :, 0] * 0.1  # Small brightness boost
    blended_lab[:, :, 0] = np.clip(frame_lab[:, :, 0] + brightness_boost * 255.0, 0, 255)
    # Blend A and B channels toward blush color - more aggressive
    blended_lab[:, :, 1] = frame_lab[:, :, 1] * (1 - intensity_map[:, :, 0]) + blush_lab_single[1] * intensity_map[:, :, 0]
    blended_lab[:, :, 2] = frame_lab[:, :, 2] * (1 - intensity_map[:, :, 0]) + blush_lab_single[2] * intensity_map[:, :, 0]
    
    blended_bgr = cv2.cvtColor(np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Soft blending to preserve skin texture while making blush very visible
    soft_mask = cv2.GaussianBlur(cheek_mask_float, (51, 51), 0)
    soft_mask_3d = np.repeat(soft_mask[:, :, None], 3, axis=2)
    
    # Final blend: subtle but visible - 50% blend for natural blush
    blend_factor = soft_mask_3d * 0.5  # Subtle but visible blend
    out = frame.astype(np.float32) * (1 - blend_factor) + blended_bgr.astype(np.float32) * blend_factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    # Add very subtle texture to prevent airbrushed look
    out = add_skin_texture(out, strength=0.5)
    
    # Optional: add soft nose glow only in warm lighting
    if light_type == "warm" and coords is not None:
        try:
            nose_x, nose_y = int(coords[1][0]), int(coords[1][1])
            h, w = frame.shape[:2]
            # Very subtle nose glow
            nose_glow = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.circle(nose_glow, (nose_x, nose_y + int(h * 0.01)),
                       int(w * 0.03), (20, 30, 70), -1)
            nose_glow = cv2.GaussianBlur(nose_glow, (25, 25), 0)
            glow_mask = (nose_glow.sum(axis=2) > 10).astype(np.float32)
            glow_mask_3d = np.repeat(glow_mask[:, :, None], 3, axis=2) * 0.08
            out = out.astype(np.float32) * (1 - glow_mask_3d) + nose_glow.astype(np.float32) * glow_mask_3d
            out = np.clip(out, 0, 255).astype(np.uint8)
        except (IndexError, KeyError):
            pass
    
    return out
