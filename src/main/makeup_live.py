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
from src.core.color_palette import (
    generate_lipstick_palette,
    generate_blush_palette,
    draw_color_palette,
    get_clicked_color
)

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

# Color picker state
selected_lipstick_color = None
selected_blush_color = None
lipstick_palette = None
blush_palette = None
lipstick_swatches = []
blush_swatches = []

# Mouse callback for color picker
def mouse_callback(event, x, y, flags, param):
    global selected_lipstick_color, selected_blush_color, lipstick_swatches, blush_swatches
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")
        # Check lipstick palette
        if lipstick_swatches:
            color, palette_type = get_clicked_color(x, y, lipstick_swatches)
            if color is not None:
                selected_lipstick_color = color
                print(f"✓ Selected lipstick color: RGB{color}")
                return
        
        # Check blush palette
        if blush_swatches:
            color, palette_type = get_clicked_color(x, y, blush_swatches)
            if color is not None:
                selected_blush_color = color
                print(f"✓ Selected blush color: RGB{color}")
                return
        
        print(f"  No color swatch clicked (lipstick swatches: {len(lipstick_swatches)}, blush swatches: {len(blush_swatches)})")

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("💄 TIVORA Dynamic AR Makeup Engine Active")
print("Click on color swatches to try different shades")
print("Press ESC to exit, 'r' to reset to auto colors")

# Set up mouse callback
cv2.namedWindow('Tivora AR Makeup - Prototype')
cv2.setMouseCallback('Tivora AR Makeup - Prototype', mouse_callback)

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
        
        # Generate color palettes based on skin tone (update when skin tone changes)
        if prev_skin_lab is not None:
            new_lipstick_palette = generate_lipstick_palette(prev_skin_lab, light_type)
            new_blush_palette = generate_blush_palette(prev_skin_lab, light_type)
            
            # Set default colors (first shade) if not already selected
            if selected_lipstick_color is None and new_lipstick_palette and len(new_lipstick_palette) > 0:
                selected_lipstick_color = new_lipstick_palette[0]
                print(f"Auto-applied default lipstick: RGB{selected_lipstick_color}")
            
            if selected_blush_color is None and new_blush_palette and len(new_blush_palette) > 0:
                selected_blush_color = new_blush_palette[0]
                print(f"Auto-applied default blush: RGB{selected_blush_color}")
            
            lipstick_palette = new_lipstick_palette
            blush_palette = new_blush_palette
        
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
        
        # Apply lipstick with selected or dynamic color
        if lip_mask is not None and lip_mask.max() > 0.01:
            out = apply_dynamic_lipstick(
                out, lip_mask, prev_skin_lab, 
                light_type=light_type, coords=coords, intensity=0.7,
                custom_color_rgb=selected_lipstick_color
            )
        
        # Apply dynamic blush with selected or dynamic color
        if cheek_mask is not None and cheek_mask.max() > 0.01:
            out = apply_dynamic_blush(
                out, cheek_mask, prev_skin_lab,
                light_type=light_type, brightness=brightness, 
                coords=coords, strength=1.0,
                custom_color_rgb=selected_blush_color
            )
        
        # Draw color picker palettes (centered below face)
        if lipstick_palette is not None:
            # Position lipstick palette in bottom area
            lipstick_swatches = draw_color_palette(out, lipstick_palette, 'lipstick', start_y=int(h * 0.70))
            for i, swatch in enumerate(lipstick_swatches):
                swatch['type'] = 'lipstick'
                # Highlight selected color with bright green border
                # Compare tuples properly
                if selected_lipstick_color is not None:
                    swatch_color = swatch['color']
                    if (isinstance(swatch_color, tuple) and isinstance(selected_lipstick_color, tuple) and
                        len(swatch_color) == 3 and len(selected_lipstick_color) == 3 and
                        swatch_color[0] == selected_lipstick_color[0] and
                        swatch_color[1] == selected_lipstick_color[1] and
                        swatch_color[2] == selected_lipstick_color[2]):
                        cv2.rectangle(out, 
                                     (swatch['x'] - 4, swatch['y'] - 4),
                                     (swatch['x'] + swatch['width'] + 4, swatch['y'] + swatch['height'] + 4),
                                     (0, 255, 0), 5)
        
        if blush_palette is not None:
            # Position blush palette below lipstick
            blush_swatches = draw_color_palette(out, blush_palette, 'blush', start_y=int(h * 0.85))
            for i, swatch in enumerate(blush_swatches):
                swatch['type'] = 'blush'
                # Highlight selected color with bright green border
                # Compare tuples properly
                if selected_blush_color is not None:
                    swatch_color = swatch['color']
                    if (isinstance(swatch_color, tuple) and isinstance(selected_blush_color, tuple) and
                        len(swatch_color) == 3 and len(selected_blush_color) == 3 and
                        swatch_color[0] == selected_blush_color[0] and
                        swatch_color[1] == selected_blush_color[1] and
                        swatch_color[2] == selected_blush_color[2]):
                        cv2.rectangle(out,
                                     (swatch['x'] - 4, swatch['y'] - 4),
                                     (swatch['x'] + swatch['width'] + 4, swatch['y'] + swatch['height'] + 4),
                                     (0, 255, 0), 5)
        
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
            
            # Show selected colors status
            y_offset = 148
            if selected_lipstick_color is not None:
                cv2.putText(out, "Lipstick: CUSTOM", (12, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(out, "Lipstick: AUTO", (12, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if selected_blush_color is not None:
                cv2.putText(out, "Blush: CUSTOM", (12, y_offset + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(out, "Blush: AUTO", (12, y_offset + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Tivora AR Makeup - Prototype', out)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    elif k == ord('r') or k == ord('R'):  # Reset to auto colors
        selected_lipstick_color = None
        selected_blush_color = None
        print("Reset to automatic color selection")

cap.release()
cv2.destroyAllWindows()
