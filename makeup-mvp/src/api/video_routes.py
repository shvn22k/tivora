# src/api/video_routes.py
"""
FastAPI routes for real-time video streaming with makeup application.
"""
import cv2
import numpy as np
import asyncio
import os
import tempfile
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse
import base64
import json
from typing import Optional
from src.services.makeup_service import makeup_service

router = APIRouter(prefix="/video", tags=["video"])


class VideoStreamManager:
    """Manages video stream connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.stream_settings = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast_frame(self, frame_data: dict):
        """Broadcast frame data to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(frame_data)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


stream_manager = VideoStreamManager()


@router.websocket("/stream")
async def video_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming with makeup.
    
    Client sends base64 encoded frames, server processes and returns them.
    """
    await stream_manager.connect(websocket)
    
    # Default settings
    settings = {
        'apply_lipstick': True,
        'apply_blush': True,
        'lipstick_color': None,
        'blush_color': None,
        'lipstick_intensity': 0.9,
        'blush_intensity': 1.0,
    }
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle settings update
            if message.get('type') == 'settings':
                settings.update(message.get('settings', {}))
                await websocket.send_json({'type': 'settings_updated', 'settings': settings})
                continue
            
            # Handle frame processing
            if message.get('type') == 'frame':
                frame_base64 = message.get('frame')
                if not frame_base64:
                    continue
                
                # Decode frame
                try:
                    frame_bytes = base64.b64decode(frame_base64)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        continue
                    
                    # Process frame with makeup
                    result = makeup_service.process_image(
                        image=frame,
                        lipstick_color=settings.get('lipstick_color'),
                        blush_color=settings.get('blush_color'),
                        apply_lipstick=settings.get('apply_lipstick', True),
                        apply_blush=settings.get('apply_blush', True),
                        lipstick_intensity=settings.get('lipstick_intensity', 0.9),
                        blush_intensity=settings.get('blush_intensity', 1.0),
                    )
                    
                    if result['success']:
                        # Encode processed frame
                        _, buffer = cv2.imencode('.jpg', result['image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                        processed_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send processed frame back
                        await websocket.send_json({
                            'type': 'frame',
                            'frame': processed_base64,
                            'metadata': result.get('metadata', {})
                        })
                    else:
                        await websocket.send_json({
                            'type': 'error',
                            'message': result.get('message', 'Processing failed')
                        })
                
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': f'Frame processing error: {str(e)}'
                    })
    
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        stream_manager.disconnect(websocket)


@router.post("/upload-stream")
async def upload_video_stream(
    file: UploadFile = File(..., description="Video file to process"),
    apply_lipstick: bool = True,
    apply_blush: bool = True,
    lipstick_r: Optional[int] = None,
    lipstick_g: Optional[int] = None,
    lipstick_b: Optional[int] = None,
    blush_r: Optional[int] = None,
    blush_g: Optional[int] = None,
    blush_b: Optional[int] = None,
    lipstick_intensity: float = 0.9,
    blush_intensity: float = 1.0,
):
    """
    Upload video file and stream processed frames.
    Returns a streaming response with processed video frames.
    """
    async def process_video_stream():
        # Read video file
        video_bytes = await file.read()
        
        # Create temporary file for video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_path = temp_file.name
            temp_file.write(video_bytes)
        
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            yield b"Error: Could not open video file"
            return
        
        # Parse colors
        lipstick_color = None
        if lipstick_r is not None and lipstick_g is not None and lipstick_b is not None:
            lipstick_color = (lipstick_r, lipstick_g, lipstick_b)
        
        blush_color = None
        if blush_r is not None and blush_g is not None and blush_b is not None:
            blush_color = (blush_r, blush_g, blush_b)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = makeup_service.process_image(
                    image=frame,
                    lipstick_color=lipstick_color,
                    blush_color=blush_color,
                    apply_lipstick=apply_lipstick,
                    apply_blush=apply_blush,
                    lipstick_intensity=lipstick_intensity,
                    blush_intensity=blush_intensity,
                )
                
                if result['success']:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', result['image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    
                    # Stream frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Small delay to control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
        
        finally:
            cap.release()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    return StreamingResponse(
        process_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/camera-stream")
async def camera_stream(
    apply_lipstick: bool = True,
    apply_blush: bool = True,
    lipstick_r: Optional[int] = None,
    lipstick_g: Optional[int] = None,
    lipstick_b: Optional[int] = None,
    blush_r: Optional[int] = None,
    blush_g: Optional[int] = None,
    blush_b: Optional[int] = None,
    lipstick_intensity: float = 0.9,
    blush_intensity: float = 1.0,
    camera_id: int = 0,
):
    """
    Stream live camera feed with makeup applied.
    Note: This uses the server's camera, not the client's.
    For client-side camera, use WebSocket endpoint.
    """
    async def generate_frames():
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            yield b"Error: Could not open camera"
            return
        
        # Parse colors
        lipstick_color = None
        if lipstick_r is not None and lipstick_g is not None and lipstick_b is not None:
            lipstick_color = (lipstick_r, lipstick_g, lipstick_b)
        
        blush_color = None
        if blush_r is not None and blush_g is not None and blush_b is not None:
            blush_color = (blush_r, blush_g, blush_b)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = makeup_service.process_image(
                    image=frame,
                    lipstick_color=lipstick_color,
                    blush_color=blush_color,
                    apply_lipstick=apply_lipstick,
                    apply_blush=apply_blush,
                    lipstick_intensity=lipstick_intensity,
                    blush_intensity=blush_intensity,
                )
                
                if result['success']:
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', result['image'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    
                    # Stream frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                await asyncio.sleep(0.033)  # ~30 FPS
        
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

