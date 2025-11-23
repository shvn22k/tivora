# Real-Time Video Streaming API

The API now supports real-time video streaming with makeup application.

## Endpoints

### 1. WebSocket Stream (Recommended for Frontend)
**WebSocket** `/video/stream`

Real-time bidirectional communication for processing video frames.

**Client → Server Messages:**
```json
// Send frame for processing
{
  "type": "frame",
  "frame": "base64_encoded_image_data"
}

// Update settings
{
  "type": "settings",
  "settings": {
    "apply_lipstick": true,
    "apply_blush": true,
    "apply_eyeshadow": false,
    "lipstick_color": [200, 50, 100],
    "blush_color": [255, 150, 150],
    "lipstick_intensity": 0.9,
    "blush_intensity": 1.0,
    "eyeshadow_intensity": 1.0
  }
}
```

**Server → Client Messages:**
```json
// Processed frame
{
  "type": "frame",
  "frame": "base64_encoded_processed_image",
  "metadata": {
    "face_detected": true,
    "skin_tone_hex": "#F5D5C8",
    "lighting_type": "warm",
    "brightness": 0.65
  }
}

// Error
{
  "type": "error",
  "message": "Error description"
}

// Settings updated
{
  "type": "settings_updated",
  "settings": {...}
}
```

### 2. Camera Stream (Server-side Camera)
**GET** `/video/camera-stream`

Stream from server's camera with makeup applied.

**Query Parameters:**
- `apply_lipstick`, `apply_blush`, `apply_eyeshadow` (bool)
- `lipstick_r`, `lipstick_g`, `lipstick_b` (int, 0-255)
- `blush_r`, `blush_g`, `blush_b` (int, 0-255)
- `lipstick_intensity`, `blush_intensity`, `eyeshadow_intensity` (float, 0.0-1.0)
- `camera_id` (int, default: 0)

**Response:** MJPEG stream (multipart/x-mixed-replace)

### 3. Video Upload Stream
**POST** `/video/upload-stream`

Upload a video file and get processed frames streamed back.

**Query Parameters:** Same as camera-stream

**Response:** MJPEG stream (multipart/x-mixed-replace)

## Frontend Integration Example

### WebSocket (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/video/stream');

ws.onopen = () => {
  console.log('Connected to video stream');
  
  // Update settings
  ws.send(JSON.stringify({
    type: 'settings',
    settings: {
      apply_lipstick: true,
      apply_blush: true,
      lipstick_intensity: 0.9,
      blush_intensity: 1.0
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'frame') {
    // Display processed frame
    const img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + data.frame;
    document.body.appendChild(img);
    
    // Or update existing image
    // document.getElementById('video').src = 'data:image/jpeg;base64,' + data.frame;
  }
};

// Send frames from video element
const video = document.getElementById('video');
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

video.addEventListener('play', () => {
  function sendFrame() {
    if (!video.paused && !video.ended) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      
      const frameData = canvas.toDataURL('image/jpeg').split(',')[1];
      ws.send(JSON.stringify({
        type: 'frame',
        frame: frameData
      }));
      
      setTimeout(sendFrame, 33); // ~30 FPS
    }
  }
  sendFrame();
});
```

### MJPEG Stream (HTML)

```html
<!-- Camera stream -->
<img src="http://localhost:8000/video/camera-stream?apply_lipstick=true&apply_blush=true" />

<!-- Video upload stream -->
<form action="http://localhost:8000/video/upload-stream" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="video/*" />
  <input type="submit" value="Upload and Stream" />
</form>
```

## Performance Notes

- WebSocket is recommended for client-side camera feeds
- Server-side camera stream uses server's camera (not client's)
- Frame rate is controlled to ~30 FPS for performance
- JPEG quality is set to 85% for balance between quality and size

