# TIVORA AR Makeup API

FastAPI endpoints for applying AR makeup to images.

## Installation

```bash
pip install -r requirements.txt
```

## Running the API

```bash
# Using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python module
python -m uvicorn src.api.main:app --reload
```

## API Endpoints

### 1. Apply Makeup (Image Response)
**POST** `/makeup/apply`

Upload an image and get back the processed image with makeup applied.

**Query Parameters:**
- `apply_lipstick` (bool): Whether to apply lipstick (default: true)
- `apply_blush` (bool): Whether to apply blush (default: true)
- `apply_eyeshadow` (bool): Whether to apply eyeshadow (default: false)
- `lipstick_r`, `lipstick_g`, `lipstick_b` (int, 0-255): Custom lipstick color
- `blush_r`, `blush_g`, `blush_b` (int, 0-255): Custom blush color
- `eyeshadow_r`, `eyeshadow_g`, `eyeshadow_b` (int, 0-255): Custom eyeshadow color
- `lipstick_intensity` (float, 0.0-1.0): Lipstick intensity (default: 0.9)
- `blush_intensity` (float, 0.0-1.0): Blush intensity (default: 1.0)
- `eyeshadow_intensity` (float, 0.0-1.0): Eyeshadow intensity (default: 1.0)

**Response:** Processed image (JPEG/PNG)

### 2. Apply Makeup (JSON Response)
**POST** `/makeup/apply-json`

Upload an image with JSON request body for more structured requests.

**Request Body:**
```json
{
  "apply_lipstick": true,
  "apply_blush": true,
  "apply_eyeshadow": false,
  "lipstick_color": {"r": 200, "g": 50, "b": 100},
  "blush_color": {"r": 255, "g": 150, "b": 150},
  "lipstick_intensity": 0.9,
  "blush_intensity": 1.0
}
```

**Response:**
```json
{
  "success": true,
  "image_base64": "...",
  "metadata": {
    "face_detected": true,
    "skin_tone_hex": "#F5D5C8",
    "lighting_type": "warm",
    "brightness": 0.65
  }
}
```

### 3. Get Color Palettes
**POST** `/makeup/palettes`

Get suggested color palettes based on detected skin tone without applying makeup.

**Response:**
```json
{
  "success": true,
  "palettes": {
    "lipstick": [[200, 50, 100], ...],
    "blush": [[255, 150, 150], ...],
    "eyeshadow": [[120, 80, 100], ...]
  },
  "skin_tone_hex": "#F5D5C8",
  "lighting_type": "warm",
  "brightness": 0.65
}
```

### 4. Health Check
**GET** `/makeup/health`

Check if the service is running.

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Usage

### Using curl

```bash
# Apply makeup
curl -X POST "http://localhost:8000/makeup/apply?apply_lipstick=true&apply_blush=true" \
  -F "file=@image.jpg" \
  --output result.jpg

# Get color palettes
curl -X POST "http://localhost:8000/makeup/palettes" \
  -F "file=@image.jpg"
```

### Using Python

```python
import requests

# Apply makeup
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/makeup/apply",
        files={"file": f},
        params={
            "apply_lipstick": True,
            "apply_blush": True,
            "lipstick_intensity": 0.9
        }
    )
    with open("result.jpg", "wb") as out:
        out.write(response.content)
```

