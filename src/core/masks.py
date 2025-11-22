# src/core/masks.py
import cv2
import numpy as np

# helper polygon mask
def polygon_mask_from_points(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask

def to_int_coords(landmarks, image_shape):
    h, w = image_shape[:2]
    return [(int(p.x * w), int(p.y * h), getattr(p, 'z', 0)) for p in landmarks]


def _valid_point(pt, w, h):
    x, y = pt[:2]
    return 0 <= x < w and 0 <= y < h


def _face_bbox(coords, w, h):
    xs = [x for x, y, *_ in coords if 0 <= x < w and 0 <= y < h]
    ys = [y for x, y, *_ in coords if 0 <= x < w and 0 <= y < h]
    if not xs or not ys:
        return None
    xmin, xmax = max(0, min(xs)), min(w - 1, max(xs))
    ymin, ymax = max(0, min(ys)), min(h - 1, max(ys))
    return xmin, ymin, xmax, ymax

# More comprehensive cheek landmarks for natural W-shape blush
LEFT_CHEEK_IDX = [205, 50, 93, 132, 58, 172, 136, 150, 116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 50]
RIGHT_CHEEK_IDX = [425, 280, 323, 361, 288, 397, 365, 379, 346, 347, 348, 349, 350, 351, 356, 372, 266, 425, 280]

def create_dynamic_blush_mask(frame, coords):
    """
    Create a dynamic, subtle blush mask covering cheeks and nose bridge in W-shape.
    Adapts to face shape, respects face boundaries, and handles glasses gracefully.
    """
    h, w = frame.shape[:2]
    if coords is None:
        return None

    try:
        # Key landmarks for dynamic blush placement
        left_cheekbone = np.array(coords[234][:2])  # Left cheekbone (cheek apple)
        right_cheekbone = np.array(coords[454][:2])  # Right cheekbone (cheek apple)
        nose_tip = np.array(coords[1][:2])
        nose_bridge = np.array(coords[6][:2])
        
        # Eye landmarks for positioning reference
        left_eye_bottom = np.array(coords[145][:2])  # Left eye bottom
        right_eye_bottom = np.array(coords[374][:2])  # Right eye bottom
        
        # Face boundary landmarks
        left_face = np.array(coords[234][:2])  # Left cheek
        right_face = np.array(coords[454][:2])  # Right cheek
        
    except (IndexError, KeyError):
        return None

    mask = np.zeros((h, w), dtype=np.float32)
    bbox = _face_bbox(coords, w, h)
    if bbox is None:
        return None

    xmin, ymin, xmax, ymax = bbox
    face_w = max(1, xmax - xmin)
    face_h = max(1, ymax - ymin)

    # Calculate dynamic blush position - on cheekbones, extending towards temples
    # Position blush lower on cheeks (away from eyelids) - on cheekbone line
    eye_level_y = int((left_eye_bottom[1] + right_eye_bottom[1]) / 2)
    cheek_level_y = int((left_cheekbone[1] + right_cheekbone[1]) / 2)
    
    # Position blush on cheekbone line, well below eyes to avoid touching eyelids
    # Use cheek level as base, add offset to position it on the cheekbone
    blush_y_base = int(cheek_level_y + face_h * 0.08)  # Lower on cheeks, away from eyes
    
    # Ensure blush stays within face bounds but well below eyes
    blush_y_base = np.clip(blush_y_base, 
                          eye_level_y + int(face_h * 0.12),  # Well below eyes
                          ymax - int(face_h * 0.15))
    
    # Calculate cheek centers - on actual cheekbones, extending outward
    # Position more outward (towards temples) for full cheek coverage
    left_cheek_x = int(np.clip(left_cheekbone[0] - int(face_w * 0.02), 
                               xmin + int(face_w * 0.05), xmax - int(face_w * 0.15)))
    right_cheek_x = int(np.clip(right_cheekbone[0] + int(face_w * 0.02), 
                                xmin + int(face_w * 0.15), xmax - int(face_w * 0.05)))
    
    # Nose bridge position
    nose_x = int(np.clip(nose_tip[0], xmin, xmax))
    nose_y = int(np.clip((nose_tip[1] + nose_bridge[1]) * 0.6, ymin, ymax))
    
    # Dynamic sizing based on face dimensions - much larger for full cheek coverage
    # Extend outward (towards temples) and cover full cheek area
    cheek_width = max(50, int(face_w * 0.30))  # Much wider - extends towards temples
    cheek_height = max(45, int(face_h * 0.20))  # Taller - covers full cheek height
    nose_width = max(30, int(face_w * 0.15))  # Larger nose bridge
    nose_height = max(25, int(face_h * 0.12))  # Taller nose bridge
    
    def add_soft_blob(canvas, center, axes, value):
        """Add a soft, blurred blob to the mask."""
        blob = np.zeros_like(canvas, dtype=np.uint8)
        cv2.ellipse(blob, center, axes, 0, 0, 360, value, -1)
        # Soft blur for natural blending
        blob = cv2.GaussianBlur(blob, (121, 121), 0)
        return np.maximum(canvas, blob.astype(np.float32) / 255.0)
    
    # Main cheek blobs - positioned on cheekbones, extending outward
    # Left cheek blob - main area
    left_center = (left_cheek_x, blush_y_base)
    mask = add_soft_blob(mask, left_center, (cheek_width, cheek_height), 255)
    
    # Right cheek blob - main area
    right_center = (right_cheek_x, blush_y_base)
    mask = add_soft_blob(mask, right_center, (cheek_width, cheek_height), 255)
    
    # Extend cheeks upward along cheekbone line (towards temples)
    # Left cheek extension - upward and outward
    left_extend_x = int(left_cheek_x - int(face_w * 0.05))  # More outward
    left_extend_y = int(blush_y_base - face_h * 0.08)  # Upward along cheekbone
    left_extend_center = (left_extend_x, left_extend_y)
    mask = add_soft_blob(mask, left_extend_center, 
                        (int(cheek_width * 0.7), int(cheek_height * 0.6)), 220)
    
    # Right cheek extension - upward and outward
    right_extend_x = int(right_cheek_x + int(face_w * 0.05))  # More outward
    right_extend_y = int(blush_y_base - face_h * 0.08)  # Upward along cheekbone
    right_extend_center = (right_extend_x, right_extend_y)
    mask = add_soft_blob(mask, right_extend_center, 
                        (int(cheek_width * 0.7), int(cheek_height * 0.6)), 220)
    
    # Nose bridge blob - connects cheeks in W shape
    nose_center = (nose_x, int(blush_y_base * 0.96))  # Slightly higher for W shape
    mask = add_soft_blob(mask, nose_center, (nose_width, nose_height), 200)
    
    # Connect cheeks to nose bridge smoothly with wider paths
    # Create connecting paths for W shape - wider for better coverage
    left_to_nose_points = np.array([
        [left_cheek_x + cheek_width // 4, blush_y_base],
        [nose_x - nose_width * 1.8, int(blush_y_base * 0.98)],
        [nose_x - nose_width // 2, int(blush_y_base * 0.96)],
        [nose_x, int(blush_y_base * 0.96)]
    ], dtype=np.int32)
    
    right_to_nose_points = np.array([
        [nose_x, int(blush_y_base * 0.96)],
        [nose_x + nose_width // 2, int(blush_y_base * 0.96)],
        [nose_x + nose_width * 1.8, int(blush_y_base * 0.98)],
        [right_cheek_x - cheek_width // 4, blush_y_base]
    ], dtype=np.int32)
    
    # Draw wider soft connecting paths
    connection_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.polylines(connection_mask, [left_to_nose_points], False, 220, int(face_w * 0.12))
    cv2.polylines(connection_mask, [right_to_nose_points], False, 220, int(face_w * 0.12))
    connection_mask = cv2.GaussianBlur(connection_mask, (151, 151), 0)
    mask = np.maximum(mask, connection_mask.astype(np.float32) / 255.0)
    
    # Add lower cheek coverage - extend downward for full cheek area
    # Left lower cheek
    left_lower_y = int(blush_y_base + face_h * 0.06)
    left_lower_center = (left_cheek_x, left_lower_y)
    mask = add_soft_blob(mask, left_lower_center, 
                        (int(cheek_width * 0.9), int(cheek_height * 0.7)), 200)
    
    # Right lower cheek
    right_lower_y = int(blush_y_base + face_h * 0.06)
    right_lower_center = (right_cheek_x, right_lower_y)
    mask = add_soft_blob(mask, right_lower_center, 
                        (int(cheek_width * 0.9), int(cheek_height * 0.7)), 200)
    
    # Final soft blur to blend everything together
    mask = cv2.GaussianBlur(mask, (161, 161), 0)
    
    # Clip to face bounding box - ensure it never goes outside face
    face_mask = np.zeros((h, w), dtype=np.float32)
    face_mask[ymin:ymax, xmin:xmax] = 1.0
    # Soften face boundary
    face_mask = cv2.GaussianBlur(face_mask, (51, 51), 0)
    mask = mask * face_mask
    
    # Soft vertical fade - wider range for full cheek coverage
    vertical_fade = np.ones((h, 1), dtype=np.float32)
    fade_range = int(face_h * 0.30)  # Much wider fade range for full cheek area
    
    for y in range(h):
        dist_from_center = abs(y - blush_y_base)
        if dist_from_center < fade_range:
            # Strong in center, gentle fade
            fade_factor = 1.0 - (dist_from_center / fade_range) * 0.25
            vertical_fade[y] = max(0.75, fade_factor)  # Keep more visible
        else:
            # Gentle fade away from center
            extra_dist = dist_from_center - fade_range
            fade_factor = 0.75 - (extra_dist / max(1, h - fade_range)) * 0.35
            vertical_fade[y] = max(0.25, fade_factor)  # Keep more visible at edges
    
    mask *= vertical_fade

    # Handle glasses - detect and reduce in bright reflection zones
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect very bright areas (glasses reflections)
    bright_zones = cv2.threshold(gray, 225, 1, cv2.THRESH_BINARY)[1]
    bright_zones = cv2.GaussianBlur(bright_zones, (101, 101), 0).astype(np.float32)
    # Reduce blush in bright zones (glasses reflections)
    mask *= (1.0 - bright_zones * 0.7)  # Strong reduction in bright zones
    
    # Strongly reduce in eye/eyelid area to avoid touching eyelids
    eye_y_top = int(min(left_eye_bottom[1], right_eye_bottom[1]) - face_h * 0.08)
    eye_y_bottom = int(max(left_eye_bottom[1], right_eye_bottom[1]) + face_h * 0.10)
    for y in range(max(0, eye_y_top), min(h, eye_y_bottom)):
        if y < h:
            # Strong fade in eye area - ensure blush doesn't touch eyelids
            dist_from_eye = abs(y - eye_y_bottom)
            fade = max(0.1, 1.0 - (dist_from_eye / max(1, eye_y_bottom - eye_y_top)) * 0.9)
            mask[y, :] *= fade  # Strong reduction in eye/eyelid area

    # Ensure subtle, natural coverage
    mask = np.clip(mask, 0.0, 0.85)  # Cap for subtlety
    return mask

def detect_glasses(frame, coords):
    """
    Detect potential glasses by analyzing brightness/reflections near eyes.
    Returns True if glasses detected.
    """
    if coords is None:
        return False
    
    h, w = frame.shape[:2]
    left_eye_pts = [coords[i][:2] for i in range(130, 144) if i < len(coords)]
    right_eye_pts = [coords[i][:2] for i in range(359, 374) if i < len(coords)]
    
    def brightness_ratio(eye_pts):
        if not eye_pts:
            return 0.0
        xs = [p[0] for p in eye_pts]
        ys = [p[1] for p in eye_pts]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))
        xmin, xmax = max(0, xmin), min(w-1, xmax)
        ymin, ymax = max(0, ymin), min(h-1, ymax)
        
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        
        roi = frame[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > 200)
        ratio = bright_pixels / (roi.shape[0] * roi.shape[1])
        return ratio
    
    left_bright = brightness_ratio(left_eye_pts)
    right_bright = brightness_ratio(right_eye_pts)
    avg_bright = (left_bright + right_bright) / 2
    
    return avg_bright > 0.08  # threshold for reflections typical of glasses


def create_eyeshadow_mask(frame, coords):
    """
    Creates a natural eyeshadow mask covering the eyelid area (between eye and eyebrow).
    Works with glasses by avoiding bright reflection zones.
    Returns float mask 0..1.
    """
    h, w = frame.shape[:2]
    if coords is None or len(coords) < 400:
        return None
    
    mask = np.zeros((h, w), dtype=np.float32)
    
    try:
        # MediaPipe face mesh landmarks for eyes and eyebrows
        # Left eye: 33-46 (outer to inner)
        # Right eye: 263-276 (outer to inner)
        # Left eyebrow: 107-116
        # Right eyebrow: 336-345
        
        # Left eye region
        left_eye_indices = list(range(33, 47))  # Left eye contour
        left_eyebrow_indices = list(range(107, 117))  # Left eyebrow
        
        # Right eye region
        right_eye_indices = list(range(263, 277))  # Right eye contour
        right_eyebrow_indices = list(range(336, 346))  # Right eyebrow
        
        def create_eye_mask(eye_indices, eyebrow_indices, side='left'):
            eye_pts = [coords[i][:2] for i in eye_indices if i < len(coords)]
            eyebrow_pts = [coords[i][:2] for i in eyebrow_indices if i < len(coords)]
            
            if len(eye_pts) < 6 or len(eyebrow_pts) < 6:
                return None
            
            # Get bounding box of eye and eyebrow
            eye_xs = [p[0] for p in eye_pts]
            eye_ys = [p[1] for p in eye_pts]
            brow_xs = [p[0] for p in eyebrow_pts]
            brow_ys = [p[1] for p in eyebrow_pts]
            
            x_min = max(0, int(min(min(eye_xs), min(brow_xs))))
            x_max = min(w-1, int(max(max(eye_xs), max(brow_xs))))
            y_top = max(0, int(min(brow_ys)))  # Top of eyebrow
            y_bottom = min(h-1, int(max(eye_ys)))  # Bottom of eye
            
            if x_max <= x_min or y_bottom <= y_top:
                return None
            
            # Create gradient mask: stronger near lash line, fades upward
            eye_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Get average eye position (lash line)
            eye_center_y = int(np.mean(eye_ys))
            eye_center_x = int(np.mean(eye_xs))
            
            # Create elliptical region for eyeshadow
            # Width: eye width + some padding
            eye_width = max(20, int((max(eye_xs) - min(eye_xs)) * 1.3))
            # Height: distance from eye to eyebrow
            eye_height = max(15, int((y_bottom - y_top) * 0.7))
            
            # Draw ellipse centered above the eye
            center_y = max(0, min(h-1, eye_center_y - int(eye_height * 0.3)))  # Slightly above eye center
            axes = (int(eye_width // 2), int(eye_height))
            center = (int(eye_center_x), int(center_y))
            
            # Draw filled ellipse
            cv2.ellipse(eye_mask, center, axes, 0, 0, 360, 255, -1)
            
            # Create vertical gradient: stronger at bottom (lash line), fades upward
            # Convert to float for gradient application
            eye_mask_float = eye_mask.astype(np.float32)
            for y in range(y_top, min(y_bottom + 1, h)):
                if y_bottom > y_top:
                    # Normalized position: 1.0 at bottom (lash line), 0.0 at top
                    norm_pos = (y_bottom - y) / (y_bottom - y_top)
                else:
                    norm_pos = 0.5
                # Stronger near lash line (higher norm_pos = stronger)
                gradient = np.power(norm_pos, 1.5)  # Exponential falloff
                eye_mask_float[y, x_min:x_max+1] *= gradient
            
            # Soft blur for natural look
            eye_mask_float = cv2.GaussianBlur(eye_mask_float, (31, 31), 0)
            eye_mask_float = np.clip(eye_mask_float / 255.0, 0.0, 0.8)  # Max intensity 0.8
            
            return eye_mask_float
        
        # Create masks for both eyes
        left_mask = create_eye_mask(left_eye_indices, left_eyebrow_indices, 'left')
        right_mask = create_eye_mask(right_eye_indices, right_eyebrow_indices, 'right')
        
        if left_mask is not None:
            mask = np.maximum(mask, left_mask)
        if right_mask is not None:
            mask = np.maximum(mask, right_mask)
        
        # Avoid glasses reflections (bright zones)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_zones = cv2.threshold(gray, 210, 1, cv2.THRESH_BINARY)[1]
        bright_zones = cv2.GaussianBlur(bright_zones, (41, 41), 0)
        bright_zones = bright_zones.astype(np.float32)
        mask *= (1.0 - bright_zones * 0.8)  # Strong fade in bright zones
        
        # Final soft blur
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = np.clip(mask, 0.0, 0.8)
        
        if mask.max() < 0.01:
            return None
        
        return mask
        
    except (IndexError, KeyError, ValueError) as e:
        return None


def smooth_mask(prev_mask, new_mask, decay=0.7):
    """
    Temporal smoothing for binary masks. Keeps masks alive briefly when detection drops.
    Returns float mask 0..1 or None.
    """
    if new_mask is not None:
        new_mask = new_mask.astype(np.float32)
        if new_mask.max() < 0.01:
            new_mask = None
    if prev_mask is not None:
        prev_mask = prev_mask.astype(np.float32)
        if prev_mask.max() < 0.01:
            prev_mask = None

    if new_mask is None and prev_mask is None:
        return None
    if new_mask is None:
        decayed = prev_mask * decay
        return None if decayed.max() < 0.01 else decayed
    if prev_mask is None:
        return new_mask
    blended = cv2.addWeighted(new_mask, 1 - decay, prev_mask, decay, 0)
    return blended


def _collect_points(coords, indices, w, h):
    pts = []
    for idx in indices:
        if idx >= len(coords):
            continue
        x, y, *_ = coords[idx]
        if _valid_point((x, y), w, h):
            pts.append([x, y])
    return pts


def _auto_lip_mask_from_parse(seg_map, coords, min_ratio, max_ratio):
    if coords is None:
        return None
    h, w = seg_map.shape
    bbox = _face_bbox(coords, w, h)
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    face_h = max(1, ymax - ymin)
    face_w = max(1, xmax - xmin)

    best = None
    best_score = 0.0
    unique_classes = np.unique(seg_map)
    for cls in unique_classes:
        if cls == 0:
            continue
        cls_mask = (seg_map == cls).astype(np.uint8)
        coverage = float(np.count_nonzero(cls_mask)) / float(cls_mask.size)
        if coverage < min_ratio or coverage > max_ratio:
            continue

        region = cls_mask[ymin:ymax, xmin:xmax]
        if np.count_nonzero(region) < 40:
            continue
        ys, xs = np.nonzero(region)
        cy = ymin + ys.mean()
        cx = xmin + xs.mean()
        norm_y = (cy - ymin) / face_h
        norm_x = abs((cx - (xmin + face_w/2)) / (face_w/2 + 1e-5))
        # Lips should sit near lower half (0.6-0.85) and fairly centered.
        y_score = np.exp(-((norm_y - 0.72)**2) / (2 * 0.07**2))
        x_score = np.exp(-((norm_x)**2) / (2 * 0.5**2))
        score = coverage * y_score * x_score
        if score > best_score:
            best_score = score
            best = cls_mask

    if best is None or best_score < 1e-4:
        return None
    return best.astype(np.uint8) * 255


def get_lip_mask_from_parse(seg_map, coords=None, min_ratio=5e-4, max_ratio=0.08):
    # default BiSeNet mapping (7 upper, 8 lower). If mask looks wrong, auto-detect.
    if seg_map is None:
        return None
    lip_mask = np.logical_or(seg_map == 7, seg_map == 8).astype(np.uint8) * 255
    coverage = float(np.count_nonzero(lip_mask > 25)) / float(lip_mask.size)
    if coverage < min_ratio or coverage > max_ratio:
        auto_mask = _auto_lip_mask_from_parse(seg_map, coords, min_ratio, max_ratio)
        if auto_mask is None:
            return None
        lip_mask = auto_mask
    lip_mask = cv2.GaussianBlur(lip_mask, (15,15), 0)
    return lip_mask.astype(np.uint8)

def get_cheek_mask_hybrid(frame, seg_map, coords, left_idx=234, right_idx=454):
    """
    Create a natural W-shape blush mask using landmarks.
    Dynamically sized based on face geometry.
    """
    h, w = frame.shape[:2]
    if coords is None:
        return None
    
    bbox = _face_bbox(coords, w, h)
    if bbox is None:
        return None
    
    xmin, ymin, xmax, ymax = bbox
    face_w = max(1, xmax - xmin)
    face_h = max(1, ymax - ymin)
    face_center_x = (xmin + xmax) / 2.0
    
    # Create separate masks for left and right cheeks
    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Left cheek
    left_pts = _collect_points(coords, LEFT_CHEEK_IDX, w, h)
    if len(left_pts) >= 4:
        left_pts_arr = np.array(left_pts, dtype=np.float32)
        # Expand left cheek region
        left_centroid = np.mean(left_pts_arr, axis=0)
        # Create elliptical region around left cheek
        left_x, left_y = int(left_centroid[0]), int(left_centroid[1])
        left_radius_x = max(15, int(face_w * 0.12))
        left_radius_y = max(12, int(face_h * 0.10))
        cv2.ellipse(left_mask, (left_x, left_y), (left_radius_x, left_radius_y), 0, 0, 360, 255, -1)
    
    # Right cheek
    right_pts = _collect_points(coords, RIGHT_CHEEK_IDX, w, h)
    if len(right_pts) >= 4:
        right_pts_arr = np.array(right_pts, dtype=np.float32)
        # Expand right cheek region
        right_centroid = np.mean(right_pts_arr, axis=0)
        # Create elliptical region around right cheek
        right_x, right_y = int(right_centroid[0]), int(right_centroid[1])
        right_radius_x = max(15, int(face_w * 0.12))
        right_radius_y = max(12, int(face_h * 0.10))
        cv2.ellipse(right_mask, (right_x, right_y), (right_radius_x, right_radius_y), 0, 0, 360, 255, -1)
    
    # Combine left and right
    mask = cv2.bitwise_or(left_mask, right_mask)
    
    if mask.max() == 0:
        return None
    
    # Constrain to cheek region (middle 40-70% of face height, avoid center nose area)
    cheek_ymin = ymin + int(face_h * 0.40)
    cheek_ymax = ymin + int(face_h * 0.70)
    constrained = np.zeros_like(mask)
    constrained[cheek_ymin:cheek_ymax, :] = mask[cheek_ymin:cheek_ymax, :]
    mask = constrained
    
    # Create W-profile: stronger on sides, weaker in center
    x = np.arange(w, dtype=np.float32)
    # Left peak (around 25% of face width from left edge)
    left_peak = xmin + face_w * 0.25
    # Right peak (around 75% of face width from left edge)
    right_peak = xmin + face_w * 0.75
    # Center dip (nose area - should be minimal)
    center = face_center_x
    
    # Create W-profile with two peaks and center dip
    left_profile = np.exp(-((x - left_peak) ** 2) / (2 * (face_w * 0.15) ** 2))
    right_profile = np.exp(-((x - right_peak) ** 2) / (2 * (face_w * 0.15) ** 2))
    center_dip = 1.0 - 0.7 * np.exp(-((x - center) ** 2) / (2 * (face_w * 0.08) ** 2))
    
    # Combine profiles for W-shape
    profile = np.maximum(left_profile, right_profile) * center_dip
    profile = np.clip(profile, 0.0, 1.0)
    
    # Apply W-profile to mask
    profile_mask = profile[None, :].repeat(h, axis=0)
    mask = mask.astype(np.float32) * profile_mask
    
    # Heavy blur for natural gradient
    blur_size = max(51, int(min(face_w, face_h) * 0.15))
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Normalize to 0-1
    mask = np.clip(mask / 255.0, 0.0, 1.0).astype(np.float32)
    
    # Validate coverage
    coverage = float(np.count_nonzero(mask > 0.01)) / float(mask.size)
    if coverage < 0.01 or coverage > 0.25:
        return None
    
    return mask

def get_lip_mask_from_landmarks(frame, coords, upper_idx_list, lower_idx_list,
                                min_ratio=8e-4, max_ratio=0.05):
    """
    Create a continuous lip mask covering both upper and lower lips.
    Expands the mask to be natural-looking, not just point-based.
    """
    if coords is None:
        return None
    h, w = frame.shape[:2]
    
    # Collect all lip points (both upper and lower)
    all_pts = []
    for idx in upper_idx_list + lower_idx_list:
        if idx >= len(coords):
            continue
        x, y, *_ = coords[idx]
        if _valid_point((x, y), w, h):
            all_pts.append([x, y])
    
    if len(all_pts) < 8:
        return None
    
    pts_arr = np.array(all_pts, dtype=np.float32)
    bbox = _face_bbox(coords, w, h)
    
    # Filter to lip region within face
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        face_h = max(1, ymax - ymin)
        lip_ymin = ymin + 0.35 * face_h
        lip_ymax = ymin + 0.85 * face_h
        mask_valid = (pts_arr[:,1] >= lip_ymin) & (pts_arr[:,1] <= lip_ymax)
        pts_arr = pts_arr[mask_valid]
    
    if pts_arr.shape[0] < 6:
        return None

    # Remove outliers
    centroid = np.mean(pts_arr, axis=0)
    dists = np.linalg.norm(pts_arr - centroid, axis=1)
    if pts_arr.shape[0] >= 8:
        thresh = max(10.0, np.median(dists) * 2.0)
        pts_arr = pts_arr[dists <= thresh]
    
    if pts_arr.shape[0] < 6:
        return None

    # Create mask from convex hull
    pts = np.round(pts_arr).astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = polygon_mask_from_points(frame.shape, hull.reshape(-1, 2))
    
    # Expand mask to make it continuous and natural (dilate then blur)
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        face_w = max(1, xmax - xmin)
        face_h = max(1, ymax - ymin)
        # Dynamic kernel size based on face size
        dilate_size = max(3, int(min(face_w, face_h) * 0.03))
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Smooth blur for natural look
    blur_size = max(15, int((h + w) / 100))
    if blur_size % 2 == 0:
        blur_size += 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Clip to face region
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        pad_x = int((xmax - xmin) * 0.2)
        pad_y = int((ymax - ymin) * 0.15)
        crop = np.zeros_like(mask)
        crop[max(ymin-pad_y,0):min(ymax+pad_y,h),
             max(xmin-pad_x,0):min(xmax+pad_x,w)] = 1
        mask = mask * crop

    coverage = float(np.count_nonzero(mask > 10)) / float(mask.size)
    if coverage < min_ratio or coverage > max_ratio:
        return None
    return (mask.astype(np.float32)/255.0)
