# src/main/makeup_live.py
import cv2
import numpy as np
import mediapipe as mp
from src.core.masks import (
    create_dynamic_blush_mask,
    to_int_coords,
    smooth_mask,
    polygon_mask_from_points,
)
from src.core.skin_tone import (
    dynamic_skin_tone_tracker,
    get_skin_color_bgr,
    detect_person_change
)
from src.core.lighting import detect_lighting
from src.core.lipstick_renderer import apply_dynamic_lipstick
from src.core.blush_generator import apply_dynamic_blush

# MediaPipe face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
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

prev_skin_lab = None
prev_lip_mask = None
prev_cheek_mask = None
skin_color_locked = False
frame_count = 0
STABLE_FRAMES = 30  # Lock color after 30 frames
last_face_detected = False
stable_skin_color_hex = None

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("💄 TIVORA Dynamic AR Makeup Engine Active\nPress ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = frame.copy()
    
    # Initialize masks
    lip_mask = prev_lip_mask
    cheek_mask = prev_cheek_mask

    # Facial landmarks
    results = face_mesh.process(rgb)
    coords = None
    face_detected = False
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        coords = to_int_coords(lm, frame.shape)
        face_detected = True

    # Detect person change: face disappeared and reappeared, or significant color change
    person_changed = False
    if face_detected and not last_face_detected:
        # Face just appeared - might be new person
        person_changed = True
    elif not face_detected and last_face_detected:
        # Face disappeared - reset tracking
        person_changed = True
        prev_skin_lab = None
        skin_color_locked = False
        frame_count = 0
        stable_skin_color_hex = None
    
    last_face_detected = face_detected

    if coords is not None:
        # Get current skin color (before locking check)
        current_skin_lab = dynamic_skin_tone_tracker(
            frame, coords, prev_skin_lab, 
            alpha=0.95,  # High alpha for stability
            locked=False  # Always track, locking is just for display
        )
        
        # Check for person change by color difference
        if prev_skin_lab is not None and current_skin_lab is not None:
            if detect_person_change(current_skin_lab, prev_skin_lab, threshold=20.0):
                person_changed = True
        
        # Reset if person changed
        if person_changed:
            prev_skin_lab = current_skin_lab
            skin_color_locked = False
            frame_count = 0
            stable_skin_color_hex = None
        else:
            prev_skin_lab = current_skin_lab
        
        # Dynamic skin tone tracking with high stability (alpha=0.95)
        # Lock color after STABLE_FRAMES to prevent fluctuation
        if not skin_color_locked:
            frame_count += 1
            if frame_count >= STABLE_FRAMES:
                skin_color_locked = True
                # Get stable hex color directly from BGR for accuracy
                _, stable_skin_color_hex = get_skin_color_bgr(frame, coords)
        
        # Detect lighting conditions
        light_type, brightness = detect_lighting(frame)
        
        # Create lip mask from landmarks
        lip_points = [coords[i][:2] for i in UPPER_LIP_IDX + LOWER_LIP_IDX[::-1] if i < len(coords)]
        if len(lip_points) >= 6:
            lip_mask_raw = polygon_mask_from_points(frame.shape, lip_points)
            lip_mask_raw = cv2.GaussianBlur(lip_mask_raw, (65, 65), 0).astype(np.float32) / 255.0
            lip_mask = smooth_mask(prev_lip_mask, lip_mask_raw, decay=0.6)
            prev_lip_mask = lip_mask
        
        # Create dynamic blush mask
        cheek_mask_raw = create_dynamic_blush_mask(frame, coords)
        cheek_mask = smooth_mask(prev_cheek_mask, cheek_mask_raw, decay=0.75)
        prev_cheek_mask = cheek_mask
        
        # Apply lipstick with dynamic color selection
        if lip_mask is not None and lip_mask.max() > 0.01:
            out = apply_dynamic_lipstick(
                out, lip_mask, prev_skin_lab, 
                light_type=light_type, coords=coords, intensity=0.7
            )
        
        # Apply dynamic blush with lighting-aware adjustments
        if cheek_mask is not None and cheek_mask.max() > 0.01:
            out = apply_dynamic_blush(
                out, cheek_mask, prev_skin_lab,
                light_type=light_type, brightness=brightness, 
                coords=coords, strength=1.0
            )
        
        # Display info with accurate hex color
        if prev_skin_lab is not None:
            # Use stable hex if locked, otherwise get current color directly from BGR
            if skin_color_locked and stable_skin_color_hex:
                hex_color = stable_skin_color_hex
            else:
                # Get color directly from BGR for accuracy
                _, hex_color = get_skin_color_bgr(frame, coords)
            
            # Display hex color prominently
            cv2.putText(out, f"Skin Color: {hex_color}", (12, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(out, f"Skin Color: {hex_color}", (12, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Additional info
            cv2.putText(out, f"Light: {light_type}", (12, 56), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(out, f"Bright: {brightness:.2f}", (12, 84), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if skin_color_locked:
                cv2.putText(out, "Color: LOCKED", (12, 112), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if person_changed:
                cv2.putText(out, "Person Changed!", (12, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Tivora AR Makeup - Prototype', out)
    
    # Debug windows
    lip_mask_vis = (np.clip(lip_mask, 0.0, 1.0) * 255).astype(np.uint8) if lip_mask is not None else np.zeros((h, w), dtype=np.uint8)
    cheek_mask_vis = (np.clip(cheek_mask, 0.0, 1.0) * 255).astype(np.uint8) if cheek_mask is not None else np.zeros((h, w), dtype=np.uint8)
    cv2.imshow("DEBUG_LIP_MASK", lip_mask_vis)
    cv2.imshow("DEBUG_CHEEK_MASK", cheek_mask_vis)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
