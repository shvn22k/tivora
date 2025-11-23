# Streamlit App for Testing TIVORA AR Makeup API

## Overview

This Streamlit app provides a user-friendly interface to test the TIVORA AR Makeup API with live video feed from your webcam.

## Features

1. **Live Video Processing**
   - Real-time camera feed
   - Side-by-side display (original vs processed)
   - Adjustable frame rate (~30 FPS)

2. **Makeup Controls**
   - Toggle lipstick, blush, and eyeshadow
   - Custom color selection (RGB sliders)
   - Intensity controls for each makeup type

3. **API Testing**
   - Health check endpoint
   - Image upload and processing
   - WebSocket connection status

4. **Settings Panel**
   - Configure API URL
   - Configure WebSocket URL
   - Real-time settings updates

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the FastAPI server is running:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Usage

1. **Start the API server** (if not already running)

2. **Launch Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Configure API URL** (if needed) in the sidebar

4. **Click "Start Camera"** to begin live processing

5. **Adjust settings** in the sidebar:
   - Toggle makeup types on/off
   - Adjust colors using RGB sliders
   - Control intensity levels

6. **View results**:
   - Left side: Original camera feed
   - Right side: Processed feed with makeup applied

7. **Test image upload**:
   - Upload an image in the sidebar
   - Click "Process Image" to test single image processing

## Features Tested

- ✅ Live video feed processing
- ✅ Real-time makeup application
- ✅ Custom color selection
- ✅ Intensity controls
- ✅ Multiple makeup types (lipstick, blush, eyeshadow)
- ✅ Health check endpoint
- ✅ Image upload processing
- ✅ API connectivity

## Troubleshooting

1. **Camera not opening**:
   - Check if camera is being used by another application
   - Try changing camera index (0 or 1)

2. **API not connecting**:
   - Verify API server is running
   - Check API URL in sidebar
   - Ensure CORS is properly configured

3. **Slow processing**:
   - Reduce frame rate
   - Lower image quality
   - Check API server performance

## Notes

- The app uses HTTP endpoint for frame processing (simpler implementation)
- For WebSocket support, you can modify the code to use the WebSocket endpoint
- Processing speed depends on API server performance and network latency

