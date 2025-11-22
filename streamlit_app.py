"""
Streamlit app for testing the TIVORA AR Makeup API using live video feed.
Replicates the behavior of makeup_live.py (skin tone display, palette suggestions).
"""
import base64
import json
import time
from io import BytesIO

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TIVORA AR Makeup Tester",
    page_icon="💄",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Sidebar configuration
# ------------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
API_BASE_URL = st.sidebar.text_input("API Base URL", value="http://localhost:8000").rstrip("/")

# ------------------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------------------
session_defaults = {
    "processing": False,
    "video_capture": None,
    "last_metadata": None,
    "lipstick_color": None,
    "blush_color": None,
    "last_error_message": None,
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
@st.cache_data(ttl=30)
def check_api_health(base_url: str) -> bool:
    """Check if the FastAPI service is reachable."""
    try:
        response = requests.get(f"{base_url}/makeup/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def build_settings_payload() -> dict:
    """Build settings payload to send to the API."""
    # Get colors from session state
    lipstick_color = st.session_state.get("lipstick_color")
    blush_color = st.session_state.get("blush_color")
    
    payload = {
        "apply_lipstick": apply_lipstick,
        "apply_blush": apply_blush,
        "lipstick_intensity": lipstick_intensity,
        "blush_intensity": blush_intensity,
    }

    # Include selected colors if available (from palette selection)
    # If None, API will auto-select first shade from palette
    if lipstick_color is not None:
        payload["lipstick_color"] = {
            "r": int(lipstick_color[0]),
            "g": int(lipstick_color[1]),
            "b": int(lipstick_color[2]),
        }

    if blush_color is not None:
        payload["blush_color"] = {
            "r": int(blush_color[0]),
            "g": int(blush_color[1]),
            "b": int(blush_color[2]),
        }

    return payload


def process_frame_via_api(frame: np.ndarray):
    """Send a frame to the API and return processed image + metadata."""
    try:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
        payload = build_settings_payload()
        data = {"settings": json.dumps(payload)}

        response = requests.post(
            f"{API_BASE_URL}/makeup/apply-json",
            files=files,
            data=data,
            timeout=15,
        )

        if response.status_code != 200:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text
            return None, {"error": error_detail}

        result = response.json()
        image_base64 = result.get("image_base64")
        metadata = result.get("metadata", {})

        if image_base64 is None:
            return None, metadata

        processed_bytes = base64.b64decode(image_base64)
        processed_array = np.frombuffer(processed_bytes, np.uint8)
        processed_image = cv2.imdecode(processed_array, cv2.IMREAD_COLOR)
        return processed_image, metadata

    except Exception as exc:
        return None, {"error": f"API unreachable: {exc}"}


def display_palette(label: str, palette: list):
    """Display color palette suggestions."""
    if not palette:
        st.sidebar.write(f"No {label.lower()} suggestions available.")
        return

    st.sidebar.markdown(f"**{label} suggestions**")
    cols = st.sidebar.columns(min(len(palette), 4))
    for idx, color in enumerate(palette):
        hex_color = "#{:02X}{:02X}{:02X}".format(*color)
        col = cols[idx % len(cols)]
        col.markdown(
            f"<div style='background:{hex_color}; padding:20px; border-radius:8px; "
            f"text-align:center; color:#000;'>{hex_color}</div>",
            unsafe_allow_html=True,
        )


def display_skin_tone(metadata: dict):
    """Display detected skin tone and brightness information."""
    if not metadata:
        # Show placeholder but don't block - allow processing to start
        st.sidebar.markdown("**Detected Skin Tone**")
        st.sidebar.info("Detecting...")
        return

    skin_hex = metadata.get("skin_tone_hex", "#FFFFFF")
    lighting = metadata.get("lighting_type", "N/A")
    brightness = metadata.get("brightness", 0.0)
    stability = metadata.get("stability_status", "detecting")

    st.sidebar.markdown("**Detected Skin Tone**")
    st.sidebar.markdown(
        f"<div style='background:{skin_hex}; padding:20px; border-radius:8px; "
        f"text-align:center; color:#000;'>{skin_hex}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.write(f"Lighting: {lighting}")
    st.sidebar.write(f"Brightness: {brightness:.2f}")
    if stability and stability != "detecting":
        status_text = "✓ Locked" if stability == "locked" else "Detecting..."
        st.sidebar.write(f"Status: {status_text}")


def display_skin_tone_in_placeholder(placeholder, metadata: dict):
    """Display detected skin tone in a placeholder for real-time updates."""
    if not metadata:
        with placeholder.container():
            st.markdown("**Detected Skin Tone**")
            st.info("Detecting...")
        return

    skin_hex = metadata.get("skin_tone_hex", "#FFFFFF")
    lighting = metadata.get("lighting_type", "N/A")
    brightness = metadata.get("brightness", 0.0)
    stability = metadata.get("stability_status", "detecting")

    with placeholder.container():
        st.markdown("**Detected Skin Tone**")
        st.markdown(
            f"<div style='background:{skin_hex}; padding:20px; border-radius:8px; "
            f"text-align:center; color:#000;'>{skin_hex}</div>",
            unsafe_allow_html=True,
        )
        st.write(f"Lighting: {lighting}")
        st.write(f"Brightness: {brightness:.2f}")
        if stability:
            if stability == "locked":
                st.success("✓ Locked")
            elif stability in ["cached", "updated"]:
                st.info("✓ Stable")
            else:
                st.info("Detecting...")


def display_palettes_in_placeholder(placeholder, metadata: dict):
    """Display palettes availability status in placeholder."""
    if not metadata:
        return
    
    lipstick_palette = metadata.get("lipstick_palette")
    blush_palette = metadata.get("blush_palette")
    
    with placeholder.container():
        if lipstick_palette:
            st.markdown(f"**✓ {len(lipstick_palette)} Lipstick Shades Available**")
        if blush_palette:
            st.markdown(f"**✓ {len(blush_palette)} Blush Shades Available**")
        if not lipstick_palette and not blush_palette:
            st.info("Shades will appear once skin tone is detected...")


# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------
api_status = check_api_health(API_BASE_URL)
if api_status:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Not Reachable")
    st.sidebar.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Makeup Types")
apply_lipstick = st.sidebar.checkbox("Apply Lipstick", value=True)
apply_blush = st.sidebar.checkbox("Apply Blush", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Color Selection")

# Initialize selected colors from session state or metadata - do this immediately
# Track if we've shown palettes before to trigger rerun on first appearance
if 'palettes_shown' not in st.session_state:
    st.session_state.palettes_shown = False

if st.session_state.last_metadata:
    # Auto-select first shade from palette immediately if not already selected
    lipstick_palette = st.session_state.last_metadata.get("lipstick_palette")
    blush_palette = st.session_state.last_metadata.get("blush_palette")
    
    # Check if palettes just became available
    palettes_available = (lipstick_palette and len(lipstick_palette) > 0) or (blush_palette and len(blush_palette) > 0)
    if palettes_available and not st.session_state.palettes_shown:
        st.session_state.palettes_shown = True
        # Trigger rerun to show palettes immediately
        if not st.session_state.processing:
            st.rerun()
    
    if st.session_state.lipstick_color is None and lipstick_palette and len(lipstick_palette) > 0:
        st.session_state.lipstick_color = tuple(lipstick_palette[0])
    
    if st.session_state.blush_color is None and blush_palette and len(blush_palette) > 0:
        st.session_state.blush_color = tuple(blush_palette[0])

# Display palette swatches for selection
# We'll render these after the "Skin Tone & Suggestions" section

st.sidebar.markdown("---")
st.sidebar.subheader("Intensity")
lipstick_intensity = st.sidebar.slider("Lipstick Intensity", 0.0, 1.0, 0.9, 0.05)
blush_intensity = st.sidebar.slider("Blush Intensity", 0.0, 1.0, 1.0, 0.05)

# ------------------------------------------------------------------------------
# Main layout
# ------------------------------------------------------------------------------
st.title("💄 TIVORA AR Makeup API Tester")
st.write("Live demo replicating the original makeup_live experience.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📹 Live Camera Feed")
    start_button = st.button("▶️ Start Camera", type="primary")
    stop_button = st.button("⏹️ Stop Camera")

with col2:
    st.subheader("🎯 Status")
    status_placeholder = st.empty()

image_placeholder = st.empty()

# ------------------------------------------------------------------------------
# Display skin tone + palettes if available - use placeholders for real-time updates
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Skin Tone & Suggestions")
skin_tone_placeholder = st.sidebar.empty()
palette_placeholder = st.sidebar.empty()

# Initial display
display_skin_tone_in_placeholder(skin_tone_placeholder, st.session_state.last_metadata)
display_palettes_in_placeholder(palette_placeholder, st.session_state.last_metadata)

# Display palette swatches for selection
st.sidebar.markdown("---")
st.sidebar.subheader("Color Selection")

# Status placeholder for swatches availability
swatch_status_placeholder = st.sidebar.empty()

# Render selectboxes at top level so they persist - they'll appear when metadata is available
if st.session_state.last_metadata:
    # Clear status message when swatches are available
    swatch_status_placeholder.empty()
    lipstick_palette = st.session_state.last_metadata.get("lipstick_palette")
    blush_palette = st.session_state.last_metadata.get("blush_palette")
    
    if lipstick_palette and len(lipstick_palette) > 0:
        st.sidebar.markdown("**Select Lipstick Shade:**")
        # Create options with color preview
        options = [f"Shade {i+1} (RGB{tuple(c)})" for i, c in enumerate(lipstick_palette[:9])]
        current_idx = 0
        if st.session_state.lipstick_color:
            # Find current selection
            for i, color in enumerate(lipstick_palette[:9]):
                color_rgb = tuple(color) if isinstance(color, (list, tuple)) else color
                if (st.session_state.lipstick_color is not None and 
                    len(st.session_state.lipstick_color) == len(color_rgb) and
                    all(abs(a - b) < 1 for a, b in zip(st.session_state.lipstick_color, color_rgb))):
                    current_idx = i
                    break
        
        # Use stable key - selectbox maintains state and stays visible
        selected = st.sidebar.selectbox("", options, index=current_idx, key="lipstick_select_swatch", label_visibility="collapsed")
        if selected:
            selected_idx = options.index(selected)
            st.session_state.lipstick_color = tuple(lipstick_palette[selected_idx])
            # Show color preview
            color_rgb = tuple(lipstick_palette[selected_idx])
            st.sidebar.markdown(
                f"<div style='background:rgb{color_rgb}; padding:10px; border-radius:4px; text-align:center;'>Selected: RGB{color_rgb}</div>",
                unsafe_allow_html=True
            )
    
    if blush_palette and len(blush_palette) > 0:
        st.sidebar.markdown("**Select Blush Shade:**")
        # Create options with color preview
        options = [f"Shade {i+1} (RGB{tuple(c)})" for i, c in enumerate(blush_palette[:9])]
        current_idx = 0
        if st.session_state.blush_color:
            # Find current selection
            for i, color in enumerate(blush_palette[:9]):
                color_rgb = tuple(color) if isinstance(color, (list, tuple)) else color
                if (st.session_state.blush_color is not None and 
                    len(st.session_state.blush_color) == len(color_rgb) and
                    all(abs(a - b) < 1 for a, b in zip(st.session_state.blush_color, color_rgb))):
                    current_idx = i
                    break
        
        # Use stable key - selectbox maintains state and stays visible
        selected = st.sidebar.selectbox("", options, index=current_idx, key="blush_select_swatch", label_visibility="collapsed")
        if selected:
            selected_idx = options.index(selected)
            st.session_state.blush_color = tuple(blush_palette[selected_idx])
            # Show color preview
            color_rgb = tuple(blush_palette[selected_idx])
            st.sidebar.markdown(
                f"<div style='background:rgb{color_rgb}; padding:10px; border-radius:4px; text-align:center;'>Selected: RGB{color_rgb}</div>",
                unsafe_allow_html=True
            )
else:
    with swatch_status_placeholder.container():
        st.info("Shades will appear once skin tone is detected...")

# ------------------------------------------------------------------------------
# Camera handling
# ------------------------------------------------------------------------------
def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Unable to access camera. Please ensure it is connected and not in use.")
        return None
    return cap


# Auto-restart processing if needed (after showing palettes)
if st.session_state.get('auto_restart_processing', False) and not st.session_state.processing:
    st.session_state.auto_restart_processing = False
    cap = start_camera()
    if cap is not None:
        st.session_state.processing = True
        st.session_state.video_capture = cap
        status_placeholder.info("🟢 Processing...")

if start_button and not st.session_state.processing:
    cap = start_camera()
    if cap is not None:
        st.session_state.processing = True
        st.session_state.video_capture = cap
        status_placeholder.info("🟢 Processing...")

if stop_button and st.session_state.processing:
    st.session_state.processing = False
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release()
    st.session_state.video_capture = None
    status_placeholder.info("⏹️ Stopped")
    image_placeholder.empty()


if st.session_state.processing and st.session_state.video_capture:
    frame_placeholder = image_placeholder.empty()
    while st.session_state.processing:
        ret, frame = st.session_state.video_capture.read()
        if not ret:
            status_placeholder.error("Camera disconnected.")
            break

        processed_frame, metadata = process_frame_via_api(frame)
        if metadata and metadata.get("error"):
            error_msg = metadata["error"]
            if error_msg != st.session_state.last_error_message:
                status_placeholder.warning(error_msg)
                st.session_state.last_error_message = error_msg
            continue

        st.session_state.last_error_message = None

        if processed_frame is not None:
            display_image = np.hstack([frame, processed_frame])
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_image_rgb, channels="RGB", use_container_width=True)

            # Update metadata immediately - this triggers UI updates
            metadata_changed = (st.session_state.last_metadata != metadata)
            st.session_state.last_metadata = metadata
            
            # Update sidebar placeholders immediately with new metadata
            display_skin_tone_in_placeholder(skin_tone_placeholder, metadata)
            display_palettes_in_placeholder(palette_placeholder, metadata)
            
            # When palettes first become available, we need ONE rerun to show selectboxes
            # Process a few frames first to ensure smooth transition
            lipstick_palette = metadata.get("lipstick_palette")
            blush_palette = metadata.get("blush_palette")
            palettes_available = (lipstick_palette and len(lipstick_palette) > 0) or (blush_palette and len(blush_palette) > 0)
            
            if palettes_available:
                if not st.session_state.get('swatches_rerun_triggered', False):
                    # Initialize frame counter for smooth transition
                    if 'rerun_frame_counter' not in st.session_state:
                        st.session_state.rerun_frame_counter = 0
                    st.session_state.rerun_frame_counter += 1
                    
                    # Process 5 frames (~0.17 seconds) before breaking for smooth transition
                    if st.session_state.rerun_frame_counter >= 5:
                        st.session_state.swatches_rerun_triggered = True
                        del st.session_state.rerun_frame_counter
                        # Break once to trigger rerun - selectboxes will appear
                        # Processing continues automatically because flags remain set
                        break
                    else:
                        # Show status while waiting
                        with swatch_status_placeholder.container():
                            st.success("✓ Shades ready! Loading dropdowns...")
                else:
                    # Already triggered, just show ready status
                    swatch_status_placeholder.empty()
            else:
                with swatch_status_placeholder.container():
                    st.info("Shades will appear once skin tone is detected...")
            
            # Auto-select colors immediately when metadata is available
            if metadata:
                lipstick_palette = metadata.get("lipstick_palette")
                blush_palette = metadata.get("blush_palette")
                
                # Auto-select first shade immediately if not selected
                if st.session_state.lipstick_color is None and lipstick_palette and len(lipstick_palette) > 0:
                    st.session_state.lipstick_color = tuple(lipstick_palette[0])
                
                if st.session_state.blush_color is None and blush_palette and len(blush_palette) > 0:
                    st.session_state.blush_color = tuple(blush_palette[0])
            
            # Show status with skin tone immediately
            skin_hex = metadata.get('skin_tone_hex', 'Detecting...')
            stability = metadata.get('stability_status', 'detecting')
            status_text = f"Skin Tone: {skin_hex}"
            if stability == "locked":
                status_text += " ✓ Locked"
            elif stability in ["cached", "updated"]:
                status_text += " ✓ Stable"
            
            status_placeholder.success(status_text)
        else:
            status_placeholder.warning("Frame could not be processed.")

        time.sleep(0.033)  # ~30 FPS

# ------------------------------------------------------------------------------
# Image upload tester
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📸 Test Image Upload")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and st.sidebar.button("Process Uploaded Image"):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
        payload = build_settings_payload()
        data = {"settings": json.dumps(payload)}

        response = requests.post(
            f"{API_BASE_URL}/makeup/apply-json",
            files=files,
            data=data,
            timeout=20,
        )

        if response.status_code == 200:
            result = response.json()
            image_base64 = result.get("image_base64")
            metadata = result.get("metadata", {})
            if image_base64:
                processed_image = Image.open(BytesIO(base64.b64decode(image_base64)))
                st.sidebar.image(processed_image, caption="Processed Image", use_container_width=True)
            st.sidebar.success("Image processed successfully.")
            st.sidebar.write(metadata)
        else:
            st.sidebar.error(f"Error: {response.status_code}")
    except Exception as exc:
        st.sidebar.error(f"Processing failed: {exc}")

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("**TIVORA AR Makeup API Tester** — Make sure the FastAPI server is running before testing.")

