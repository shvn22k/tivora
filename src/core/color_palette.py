# src/core/color_palette.py
"""
Generate skin-tone appropriate lipstick and blush color palettes.
"""
import numpy as np
import cv2


def generate_lipstick_palette(skin_lab, light_type='neutral'):
    """
    Generate a palette of lipstick shades appropriate for the user's skin tone.
    Returns list of RGB tuples.
    """
    if skin_lab is None:
        # Default palette for unknown skin tone
        return [
            (220, 150, 180),  # Soft rose
            (200, 100, 130),  # Coral pink
            (180, 80, 100),   # Warm nude
            (160, 60, 80),    # Mauve
            (140, 40, 60),    # Berry
            (120, 30, 50),    # Wine
        ]
    
    skin_L, skin_A, skin_B = skin_lab
    
    # Base shades adjusted for skin tone
    base_shades = []
    
    # For fair to medium skin (L > 130)
    if skin_L > 130:
        base_shades = [
            (220, 150, 180),  # Soft rose
            (200, 120, 150),  # Pink coral
            (190, 100, 130),  # Peach nude
            (180, 90, 120),   # Warm pink
            (170, 80, 110),   # Rosewood
            (160, 70, 100),   # Dusty rose
        ]
    # For medium to tan skin (100 < L <= 130)
    elif skin_L > 100:
        base_shades = [
            (200, 110, 140),  # Warm nude
            (180, 90, 120),   # Terracotta
            (170, 80, 110),   # Mauve
            (160, 70, 100),   # Berry
            (150, 60, 90),    # Deep rose
            (140, 50, 80),    # Burgundy
        ]
    # For deep skin (L <= 100)
    else:
        base_shades = [
            (180, 80, 110),   # Rich mauve
            (160, 70, 100),   # Deep berry
            (150, 60, 90),    # Wine
            (140, 50, 80),    # Burgundy
            (130, 40, 70),    # Deep plum
            (120, 30, 60),    # Dark wine
        ]
    
    # Adjust based on lighting
    adjusted_shades = []
    for shade in base_shades:
        r, g, b = shade
        if light_type == "warm":
            r = min(255, r + 15)
            b = max(0, b - 10)
        elif light_type == "cool":
            r = max(0, r - 10)
            b = min(255, b + 15)
        adjusted_shades.append((r, g, b))
    
    return adjusted_shades


def generate_blush_palette(skin_lab, light_type='neutral'):
    """
    Generate a palette of blush shades appropriate for the user's skin tone.
    Returns list of RGB tuples.
    """
    if skin_lab is None:
        # Default palette
        return [
            (255, 200, 200),  # Soft pink
            (255, 180, 180),  # Rose
            (255, 160, 160),  # Coral
            (240, 150, 150),  # Peach
            (230, 140, 140),  # Warm pink
            (220, 130, 130),  # Dusty rose
        ]
    
    skin_L, skin_A, skin_B = skin_lab
    
    # Base blush shades
    base_shades = []
    
    # For fair to medium skin
    if skin_L > 130:
        base_shades = [
            (255, 200, 200),  # Soft pink
            (255, 180, 180),  # Rose
            (255, 160, 160),  # Coral
            (240, 150, 150),  # Peach
            (230, 140, 140),  # Warm pink
            (220, 130, 130),  # Dusty rose
        ]
    # For medium to tan skin
    elif skin_L > 100:
        base_shades = [
            (240, 160, 160),  # Warm peach
            (230, 150, 150),  # Terracotta
            (220, 140, 140),  # Coral
            (210, 130, 130),  # Rosewood
            (200, 120, 120),  # Mauve
            (190, 110, 110),  # Berry
        ]
    # For deep skin
    else:
        base_shades = [
            (220, 140, 140),  # Rich rose
            (210, 130, 130),  # Deep coral
            (200, 120, 120),  # Berry
            (190, 110, 110),  # Wine
            (180, 100, 100),  # Burgundy
            (170, 90, 90),    # Deep plum
        ]
    
    # Adjust based on lighting
    adjusted_shades = []
    for shade in base_shades:
        r, g, b = shade
        if light_type == "warm":
            r = min(255, r + 10)
            b = max(0, b - 5)
        elif light_type == "cool":
            r = max(0, r - 5)
            b = min(255, b + 10)
        adjusted_shades.append((r, g, b))
    
    return adjusted_shades


def draw_color_palette(frame, colors, palette_type='lipstick', start_x=None, start_y=None):
    """
    Draw a color palette on the frame, centered below the face area.
    Returns the palette region coordinates for click detection.
    """
    h, w = frame.shape[:2]
    
    # Larger swatches for better visibility
    swatch_size = 60
    swatch_spacing = 15
    palette_width = len(colors) * (swatch_size + swatch_spacing) - swatch_spacing
    palette_height = swatch_size + 50  # Extra space for label
    
    # Center the palette horizontally, position below face (bottom 20% of frame)
    if start_x is None:
        start_x = (w - palette_width) // 2
    if start_y is None:
        start_y = int(h * 0.75)  # Position in bottom 25% of frame
    
    # Draw background with semi-transparent overlay effect
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                  (start_x - 10, start_y - 35), 
                  (start_x + palette_width + 10, start_y + palette_height + 10),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, 
                  (start_x - 10, start_y - 35), 
                  (start_x + palette_width + 10, start_y + palette_height + 10),
                  (255, 255, 255), 3)
    
    # Draw label
    label = f"{palette_type.capitalize()} Shades:"
    cv2.putText(frame, label, (start_x, start_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw color swatches
    swatch_coords = []
    for i, color in enumerate(colors):
        x = start_x + i * (swatch_size + swatch_spacing)
        y = start_y
        
        # Convert RGB to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0])
        
        # Draw swatch with shadow effect
        cv2.rectangle(frame, (x + 2, y + 2), (x + swatch_size + 2, y + swatch_size + 2),
                     (0, 0, 0), -1)  # Shadow
        cv2.rectangle(frame, (x, y), (x + swatch_size, y + swatch_size),
                     bgr_color, -1)
        cv2.rectangle(frame, (x, y), (x + swatch_size, y + swatch_size),
                     (255, 255, 255), 3)
        
        swatch_coords.append({
            'x': x,
            'y': y,
            'width': swatch_size,
            'height': swatch_size,
            'color': color
        })
    
    return swatch_coords


def get_clicked_color(mouse_x, mouse_y, swatch_coords):
    """
    Check if mouse click is on a color swatch and return the color.
    Returns (color, palette_type) or (None, None) if no match.
    """
    if not swatch_coords:
        return None, None
    
    for swatch in swatch_coords:
        if (swatch['x'] <= mouse_x <= swatch['x'] + swatch['width'] and
            swatch['y'] <= mouse_y <= swatch['y'] + swatch['height']):
            return swatch['color'], swatch.get('type', 'lipstick')
    return None, None

