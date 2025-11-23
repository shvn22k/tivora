"""
Blending utilities: overlay-style and LAB blends.
"""
import numpy as np
import cv2


def lab_mix(frame, overlay_bgr, mask_float, intensity=0.7):
    """
    Perceptual blend in LAB color space using mask_float (0..1).
    """
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    overlay_lab = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask3 = np.repeat(mask_float[:,:,None], 3, axis=2)
    blended_lab = (1 - mask3*intensity) * frame_lab + (mask3*intensity) * overlay_lab
    blended_bgr = cv2.cvtColor(np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out = frame.copy()
    out[mask_float > 0.01] = blended_bgr[mask_float > 0.01]
    return out

def apply_masked_overlay(base, overlay_bgr, mask_float, intensity=1.0):
    """Blend overlay onto base using mask_float and lab mixing; returns BGR"""
    return lab_mix(base, overlay_bgr, mask_float, intensity=intensity)

def overlay_blend(base, overlay):
    """
    Photoshop-style overlay blend mode.
    """
    base = base.astype(np.float32)
    overlay = overlay.astype(np.float32)
    result = np.where(base < 128,
                      2 * base * overlay / 255.0,
                      255 - 2 * (255 - base) * (255 - overlay) / 255.0)
    return np.clip(result, 0, 255).astype(np.uint8)

def add_skin_texture(base, strength=1.5):
    """
    Add subtle random texture so makeup doesn't look airbrushed.
    """
    noise = np.random.normal(0, strength, base.shape[:2]).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (7, 7), 0)
    texture = np.repeat(texture[:, :, None], 3, axis=2)
    return np.clip(base.astype(np.float32) + texture, 0, 255).astype(np.uint8)

