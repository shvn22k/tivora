# src/core/skin_tone.py
import numpy as np
import cv2

def extract_skin_region_mean(frame, parse_mask):
    """
    parse_mask: class map or None
    returns (lab_triplet_float32, hex_color_string)
    """
    h,w = frame.shape[:2]
    if parse_mask is None:
        # fallback: central patch
        patch = frame[h//3:2*h//3, w//3:2*w//3]
        bgr = np.median(patch.reshape(-1,3), axis=0).astype(np.uint8)
    else:
        skin_pixels = frame[parse_mask == 1]
        if len(skin_pixels) < 50:
            # fallback center
            patch = frame[h//3:2*h//3, w//3:2*w//3]
            bgr = np.median(patch.reshape(-1,3), axis=0).astype(np.uint8)
        else:
            bgr = np.median(skin_pixels, axis=0).astype(np.uint8)
    bgr_patch = np.uint8([[bgr]])
    lab_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    hex_color = '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return lab_patch, hex_color

def smooth_skin_lab(prev_lab, current_lab, alpha=0.75):
    if current_lab is None:
        return prev_lab
    current_lab = np.array(current_lab, dtype=np.float32)
    if prev_lab is None:
        return current_lab
    prev = np.array(prev_lab, dtype=np.float32)
    return prev * alpha + current_lab * (1.0 - alpha)

def estimate_undertone(lab):
    # simple heuristic using A/B channels
    if lab is None:
        return 'neutral'
    a = lab[1]
    b = lab[2]
    if b > a + 6:
        return 'warm'
    if a > b + 6:
        return 'cool'
    return 'neutral'

def dynamic_skin_tone_tracker(frame, coords, prev_lab=None, alpha=0.95, locked=False):
    """
    Continuously update average skin tone from multiple face regions.
    Uses landmarks to sample from well-lit areas (forehead, cheeks).
    With high alpha (0.95) for stable color that doesn't fluctuate.
    
    Args:
        frame: BGR image
        coords: landmark coordinates
        prev_lab: previous LAB value for temporal smoothing
        alpha: smoothing factor (0.95 = very stable, 0.7 = more responsive)
        locked: if True, return prev_lab without updating
    """
    if locked and prev_lab is not None:
        return prev_lab
    
    if coords is None:
        return prev_lab
    
    h, w = frame.shape[:2]
    region_indices = [10, 234, 454]  # forehead, left & right cheeks
    all_pixels = []

    for idx in region_indices:
        if idx >= len(coords):
            continue
        x, y = coords[idx][:2]
        x, y = int(x), int(y)

        patch_size = 20
        y1, y2 = max(0, y-patch_size), min(h, y+patch_size)
        x1, x2 = max(0, x-patch_size), min(w, x+patch_size)
        patch = frame[y1:y2, x1:x2]

        if patch.size > 0:
            pixels = patch.reshape(-1, 3)
            brightness = np.mean(pixels, axis=1)
            bright_pixels = pixels[brightness > 50]  # drop deep shadows
            if len(bright_pixels) > 0:
                all_pixels.append(bright_pixels)

    if not all_pixels:
        return prev_lab

    all_pixels = np.vstack(all_pixels)
    bgr_color = np.median(all_pixels, axis=0).astype(np.uint8)
    
    # Convert to LAB
    bgr_patch = np.uint8([[bgr_color]])
    lab_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    
    if prev_lab is None:
        return lab_patch
    
    # Strong temporal smoothing for stable color
    return alpha * prev_lab + (1 - alpha) * lab_patch

def lab_to_rgb_hex(lab):
    """
    Convert LAB color to RGB hex string.
    Properly converts LAB -> BGR -> RGB -> HEX.
    """
    if lab is None:
        return "#000000"
    
    # Convert LAB to BGR (OpenCV uses BGR)
    lab_array = np.uint8([[np.clip(lab, 0, 255)]])
    bgr = cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)[0, 0]
    
    # BGR to RGB (swap channels)
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    
    # Clamp to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    # Convert to hex
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def get_skin_color_bgr(frame, coords):
    """
    Get skin color directly from BGR frame using landmarks.
    Samples from well-lit areas and uses brightest representative pixels.
    Returns BGR tuple and RGB hex string.
    """
    if coords is None:
        return None, "#000000"
    
    h, w = frame.shape[:2]
    region_indices = [10, 234, 454]  # forehead, left & right cheeks
    all_pixels = []

    for idx in region_indices:
        if idx >= len(coords):
            continue
        x, y = coords[idx][:2]
        x, y = int(x), int(y)

        patch_size = 20
        y1, y2 = max(0, y-patch_size), min(h, y+patch_size)
        x1, x2 = max(0, x-patch_size), min(w, x+patch_size)
        patch = frame[y1:y2, x1:x2]

        if patch.size > 0:
            pixels = patch.reshape(-1, 3)
            brightness = np.mean(pixels, axis=1)
            bright_pixels = pixels[brightness > 50]
            if len(bright_pixels) > 0:
                all_pixels.append(bright_pixels)

    if not all_pixels:
        return None, "#000000"

    all_pixels = np.vstack(all_pixels)
    bgr_color = np.median(all_pixels, axis=0).astype(np.uint8)
    
    # Convert BGR to RGB hex
    r, g, b = int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    
    return bgr_color, hex_color

def detect_person_change(current_lab, prev_lab, threshold=15.0):
    """
    Detect if a different person is in frame by checking for significant color change.
    Returns True if person likely changed.
    """
    if prev_lab is None or current_lab is None:
        return False
    
    # Calculate Euclidean distance in LAB color space
    color_diff = np.linalg.norm(np.array(current_lab) - np.array(prev_lab))
    return color_diff > threshold





def lab_to_hex(lab):
    """Convert LAB → RGB → HEX for storing in DB."""
    patch = np.zeros((1,1,3), dtype=np.uint8)
    patch[0,0,:] = np.clip(lab, 0, 255)
    rgb = cv2.cvtColor(patch, cv2.COLOR_LAB2RGB)[0,0]
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])