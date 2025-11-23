# src/main/makeup_live.py
import cv2
import numpy as np
import mediapipe as mp
from src.core.masks import (
    create_dynamic_blush_mask,
    create_eyeshadow_mask,
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
from src.core.eyeshadow_generator import apply_dynamic_eyeshadow
from src.core.color_palette import (
    generate_lipstick_palette,
    generate_blush_palette,
    generate_eyeshadow_palette,
    draw_color_palette,
    draw_vertical_collapsible_palette,
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
prev_eyeshadow_mask = None
prev_coords = None  # Store previous coords for frame skipping
skin_color_locked = False
frame_count = 0
STABLE_FRAMES = 30  # Lock color after 30 frames
last_face_detected = False
stable_skin_color_hex = None

# Color picker state
selected_lipstick_color = None
selected_blush_color = None
selected_eyeshadow_color = None
lipstick_palette = None
blush_palette = None
eyeshadow_palette = None
lipstick_swatches = []
blush_swatches = []
eyeshadow_swatches = []

# Collapsible palette state
lipstick_palette_open = False
blush_palette_open = False
lipstick_scroll_offset = 0
blush_scroll_offset = 0
MAX_SHADES_VISIBLE = 8  # Number of shades visible at once

# Store palette header positions for click detection
lipstick_header_pos = None
blush_header_pos = None

# Mouse callback for color picker
def mouse_callback(event, x, y, flags, param):
    global selected_lipstick_color, selected_blush_color, selected_eyeshadow_color
    global lipstick_swatches, blush_swatches, eyeshadow_swatches
    global lipstick_palette_open, blush_palette_open
    global lipstick_scroll_offset, blush_scroll_offset
    global lipstick_header_pos, blush_header_pos
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")
        # Only check if click is in UI panel (left side)
        if x < UI_PANEL_WIDTH:
            # Check for header clicks first (to toggle palettes)
            # Check lipstick header
            if lipstick_header_pos is not None:
                hx, hy, hw, hh = lipstick_header_pos
                if hx <= x <= hx + hw and hy <= y <= hy + hh:
                    lipstick_palette_open = not lipstick_palette_open
                    lipstick_scroll_offset = 0  # Reset scroll when toggling
                    print(f"✓ Toggled lipstick palette: {'OPEN' if lipstick_palette_open else 'CLOSED'}")
                    return
            
            # Check blush header
            if blush_header_pos is not None:
                hx, hy, hw, hh = blush_header_pos
                if hx <= x <= hx + hw and hy <= y <= hy + hh:
                    blush_palette_open = not blush_palette_open
                    blush_scroll_offset = 0  # Reset scroll when toggling
                    print(f"✓ Toggled blush palette: {'OPEN' if blush_palette_open else 'CLOSED'}")
                    return
            
            # Check for color swatch clicks
            # Check lipstick swatches (only if open)
            if lipstick_palette_open and lipstick_swatches:
                color, palette_type = get_clicked_color(x, y, lipstick_swatches)
                if color is not None:
                    selected_lipstick_color = color
                    print(f"✓ Selected lipstick color: RGB{color}")
                    return
            
            # Check blush swatches (only if open)
            if blush_palette_open and blush_swatches:
                color, palette_type = get_clicked_color(x, y, blush_swatches)
                if color is not None:
                    selected_blush_color = color
                    print(f"✓ Selected blush color: RGB{color}")
                    return

# UI Panel width (left side)
UI_PANEL_WIDTH = 400

# Try camera device 0 first, fallback to 1 if available
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("💄 TIVORA Dynamic AR Makeup Engine Active")
print("Controls:")
print("  - Click palette headers or press 'L'/'B' to toggle lipstick/blush palettes")
print("  - Click color swatches to select shades")
print("  - Press 'W'/'S' to scroll through shades (when palette is open)")
print("  - Press 'R' to reset to auto colors")
print("  - Press ESC to exit")

# Set up windows
cv2.namedWindow('Tivora AR Makeup - Prototype')
cv2.setMouseCallback('Tivora AR Makeup - Prototype', mouse_callback)
# cv2.namedWindow('Debug: Eyeshadow Mask', cv2.WINDOW_NORMAL)  # Disabled for testing

# Performance optimization: process every Nth frame
frame_skip = 2
frame_counter = 0

# UI Panel width (left side)
UI_PANEL_WIDTH = 400

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    process_frame = (frame_counter % frame_skip == 0)
    
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = frame.copy()
    
    # Create split screen: left panel for UI, right panel for camera
    display_frame = np.zeros((h, w + UI_PANEL_WIDTH, 3), dtype=np.uint8)
    display_frame[:, UI_PANEL_WIDTH:] = out  # Right side: camera feed (will be updated with makeup)
    
    # Initialize masks (use previous if available)
    lip_mask = prev_lip_mask
    cheek_mask = prev_cheek_mask
    eyeshadow_mask = prev_eyeshadow_mask

    # Facial landmarks (only process if needed for this frame)
    if process_frame:
        results = face_mesh.process(rgb)
    else:
        # Reuse previous results for skipped frames - keep using cached coords
        results = None
    
    coords = None
    face_detected = False
    if process_frame:
        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = to_int_coords(lm, frame.shape)
            prev_coords = coords  # Store for next frame
            face_detected = True
        else:
            prev_coords = None
    else:
        # Use cached coords from previous frame
        coords = prev_coords
        if coords is not None:
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
            new_eyeshadow_palette = generate_eyeshadow_palette(prev_skin_lab, light_type)
            
            # Set default colors (first shade) if not already selected
            if selected_lipstick_color is None and new_lipstick_palette and len(new_lipstick_palette) > 0:
                selected_lipstick_color = new_lipstick_palette[0]
                print(f"Auto-applied default lipstick: RGB{selected_lipstick_color}")
            
            if selected_blush_color is None and new_blush_palette and len(new_blush_palette) > 0:
                selected_blush_color = new_blush_palette[0]
                print(f"Auto-applied default blush: RGB{selected_blush_color}")
            
            # Eyeshadow auto-apply disabled for testing
            # if selected_eyeshadow_color is None and new_eyeshadow_palette and len(new_eyeshadow_palette) > 0:
            #     selected_eyeshadow_color = new_eyeshadow_palette[0]
            #     print(f"Auto-applied default eyeshadow: RGB{selected_eyeshadow_color}")
            
            lipstick_palette = new_lipstick_palette
            blush_palette = new_blush_palette
            eyeshadow_palette = new_eyeshadow_palette
        
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
        
        # Create eyeshadow mask (DISABLED for testing)
        # eyeshadow_mask_raw = create_eyeshadow_mask(frame, coords)
        # eyeshadow_mask = smooth_mask(prev_eyeshadow_mask, eyeshadow_mask_raw, decay=0.7)
        # prev_eyeshadow_mask = eyeshadow_mask
        eyeshadow_mask = None  # Disabled for now
        
        # Apply lipstick with selected or dynamic color
        if lip_mask is not None and lip_mask.max() > 0.01:
            out = apply_dynamic_lipstick(
                out, lip_mask, prev_skin_lab, 
                light_type=light_type, coords=coords, intensity=0.9,
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
        
        # Apply eyeshadow with selected or dynamic color (DISABLED for testing)
        # if eyeshadow_mask is not None and eyeshadow_mask.max() > 0.01:
        #     out = apply_dynamic_eyeshadow(
        #         out, eyeshadow_mask, prev_skin_lab,
        #         light_type=light_type, brightness=brightness,
        #         coords=coords, strength=1.0,
        #         custom_color_rgb=selected_eyeshadow_color
        #     )
        
        # Draw UI panel on left side
        ui_panel = np.zeros((h, UI_PANEL_WIDTH, 3), dtype=np.uint8)
        ui_panel.fill(40)  # Dark gray background
        
        # Draw title
        cv2.putText(ui_panel, "TIVORA AR MAKEUP", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw skin tone info
        y_pos = 70
        if prev_skin_lab is not None:
            if skin_color_locked and stable_skin_color_hex:
                hex_color = stable_skin_color_hex
            else:
                _, hex_color = get_skin_color_bgr(frame, coords)
            
            # Skin color display with color swatch
            cv2.putText(ui_panel, "Skin Tone:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(ui_panel, hex_color, (10, y_pos + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw color swatch
            try:
                r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
                cv2.rectangle(ui_panel, (150, y_pos - 15), (200, y_pos + 10), (b, g, r), -1)
                cv2.rectangle(ui_panel, (150, y_pos - 15), (200, y_pos + 10), (255, 255, 255), 2)
            except:
                pass
            
            y_pos += 50
            
            # Lighting info
            cv2.putText(ui_panel, f"Lighting: {light_type.upper()}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(ui_panel, f"Brightness: {brightness:.2f}", (10, y_pos + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if skin_color_locked:
                cv2.putText(ui_panel, "LOCKED", (10, y_pos + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            y_pos += 80
        
        # Draw color palettes on left panel (side by side, 50/50 split)
        palette_y = 150  # Start position for palettes
        palette_spacing = 5  # Small gap between palettes
        # Ensure palettes fit within UI panel with margins
        palette_width = (UI_PANEL_WIDTH - palette_spacing - 10) // 2  # 50% width minus margins
        
        eyeshadow_swatches = []  # Disabled for now
        
        # Lipstick palette (left side, 50% width)
        if lipstick_palette is not None:
            lipstick_start_x = 5  # Small margin from left edge
            lipstick_swatches, header_click, palette_height = draw_vertical_collapsible_palette(
                ui_panel, lipstick_palette, 'lipstick',
                is_open=lipstick_palette_open,
                start_x=lipstick_start_x, start_y=palette_y,
                max_shades=MAX_SHADES_VISIBLE,
                scroll_offset=lipstick_scroll_offset,
                palette_width=palette_width
            )
            
            # Store header position for click detection
            lipstick_header_pos = (header_click['x'], header_click['y'], 
                                  header_click['width'], header_click['height'])
            
            # Highlight selected color
            for swatch in lipstick_swatches:
                if selected_lipstick_color is not None:
                    swatch_color = swatch['color']
                    if (isinstance(swatch_color, tuple) and isinstance(selected_lipstick_color, tuple) and
                        len(swatch_color) == 3 and len(selected_lipstick_color) == 3 and
                        swatch_color[0] == selected_lipstick_color[0] and
                        swatch_color[1] == selected_lipstick_color[1] and
                        swatch_color[2] == selected_lipstick_color[2]):
                        cv2.rectangle(ui_panel, 
                                     (swatch['x'] - 3, swatch['y'] - 3),
                                     (swatch['x'] + swatch['width'] + 3, swatch['y'] + swatch['height'] + 3),
                                     (0, 255, 0), 3)
        
        # Blush palette (right side, 50% width, adjacent to lipstick)
        if blush_palette is not None:
            blush_start_x = 5 + palette_width + palette_spacing  # Right next to lipstick palette
            blush_swatches, header_click, palette_height = draw_vertical_collapsible_palette(
                ui_panel, blush_palette, 'blush',
                is_open=blush_palette_open,
                start_x=blush_start_x, start_y=palette_y,
                max_shades=MAX_SHADES_VISIBLE,
                scroll_offset=blush_scroll_offset,
                palette_width=palette_width
            )
            
            # Store header position for click detection
            blush_header_pos = (header_click['x'], header_click['y'], 
                                header_click['width'], header_click['height'])
            
            # Highlight selected color
            for swatch in blush_swatches:
                if selected_blush_color is not None:
                    swatch_color = swatch['color']
                    if (isinstance(swatch_color, tuple) and isinstance(selected_blush_color, tuple) and
                        len(swatch_color) == 3 and len(selected_blush_color) == 3 and
                        swatch_color[0] == selected_blush_color[0] and
                        swatch_color[1] == selected_blush_color[1] and
                        swatch_color[2] == selected_blush_color[2]):
                        cv2.rectangle(ui_panel,
                                     (swatch['x'] - 3, swatch['y'] - 3),
                                     (swatch['x'] + swatch['width'] + 3, swatch['y'] + swatch['height'] + 3),
                                     (0, 255, 0), 3)
        
        # Combine UI panel and camera feed
        display_frame[:, :UI_PANEL_WIDTH] = ui_panel
        display_frame[:, UI_PANEL_WIDTH:] = out
        
        # Show debug eyeshadow mask (DISABLED for testing)
        # if eyeshadow_mask is not None and eyeshadow_mask.max() > 0.01:
        #     debug_mask = (eyeshadow_mask * 255).astype(np.uint8)
        #     debug_mask_colored = cv2.applyColorMap(debug_mask, cv2.COLORMAP_JET)
        #     cv2.imshow('Debug: Eyeshadow Mask', debug_mask_colored)
    else:
        # No face detected - still show UI panel
        ui_panel = np.zeros((h, UI_PANEL_WIDTH, 3), dtype=np.uint8)
        ui_panel.fill(40)
        cv2.putText(ui_panel, "TIVORA AR MAKEUP", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ui_panel, "No face detected", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        display_frame[:, :UI_PANEL_WIDTH] = ui_panel
        display_frame[:, UI_PANEL_WIDTH:] = out

    cv2.imshow('Tivora AR Makeup - Prototype', display_frame)
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    elif k == ord('r') or k == ord('R'):  # Reset to auto colors
        selected_lipstick_color = None
        selected_blush_color = None
        selected_eyeshadow_color = None
        print("Reset to automatic color selection")
    elif k == ord('l') or k == ord('L'):  # Toggle lipstick palette
        lipstick_palette_open = not lipstick_palette_open
        lipstick_scroll_offset = 0
        print(f"Lipstick palette: {'OPEN' if lipstick_palette_open else 'CLOSED'}")
    elif k == ord('b') or k == ord('B'):  # Toggle blush palette
        blush_palette_open = not blush_palette_open
        blush_scroll_offset = 0
        print(f"Blush palette: {'OPEN' if blush_palette_open else 'CLOSED'}")
    elif k == ord('w') or k == ord('W'):  # Scroll up (if palette open)
        if lipstick_palette_open and lipstick_palette and lipstick_scroll_offset > 0:
            lipstick_scroll_offset = max(0, lipstick_scroll_offset - 1)
            print(f"Lipstick scroll: {lipstick_scroll_offset}")
        elif blush_palette_open and blush_palette and blush_scroll_offset > 0:
            blush_scroll_offset = max(0, blush_scroll_offset - 1)
            print(f"Blush scroll: {blush_scroll_offset}")
    elif k == ord('s') or k == ord('S'):  # Scroll down (if palette open)
        if lipstick_palette_open and lipstick_palette:
            max_scroll = max(0, len(lipstick_palette) - MAX_SHADES_VISIBLE)
            if lipstick_scroll_offset < max_scroll:
                lipstick_scroll_offset = min(max_scroll, lipstick_scroll_offset + 1)
                print(f"Lipstick scroll: {lipstick_scroll_offset}")
        elif blush_palette_open and blush_palette:
            max_scroll = max(0, len(blush_palette) - MAX_SHADES_VISIBLE)
            if blush_scroll_offset < max_scroll:
                blush_scroll_offset = min(max_scroll, blush_scroll_offset + 1)
                print(f"Blush scroll: {blush_scroll_offset}")

cap.release()
cv2.destroyAllWindows()
