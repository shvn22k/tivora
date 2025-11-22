# VTO API - Simple Version

One file, just makes Replicate API calls.

## Setup

```bash
# Install
pip install -r requirements.txt

# Set token
export REPLICATE_API_TOKEN=your_token_here

# Run
python main.py
```

## Usage

```bash
# Send batch request
curl -X POST http://localhost:8000/api/v1/tryon/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "user_image_url": "https://example.com/user.jpg",
    "garments": [
      {
        "garment_id": "g1",
        "garment_image_url": "https://example.com/garment1.jpg",
        "category": "upper_body"
      }
    ]
  }'
```

## Config

Set via environment variables:
- `REPLICATE_API_TOKEN` (required)
- `MAX_CONCURRENT` (default: 10)

## API Docs

http://localhost:8000/docs

