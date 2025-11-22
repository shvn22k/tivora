# TIVORA AR Makeup API Setup

## Overview

The makeup feature has been converted into FastAPI endpoints for frontend integration. The API provides endpoints for applying AR makeup to images.

## Structure

```
src/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── makeup_routes.py     # Makeup API endpoints
│   ├── schemas.py           # Pydantic models
│   └── README.md            # API documentation
├── services/
│   ├── __init__.py
│   └── makeup_service.py    # Core makeup processing logic
└── ...
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. `/makeup/apply` (POST)
Apply makeup to an uploaded image. Returns processed image.

### 2. `/makeup/apply-json` (POST)
Apply makeup with JSON request body. Returns base64 encoded image + metadata.

### 3. `/makeup/palettes` (POST)
Get color palettes based on skin tone without applying makeup.

### 4. `/makeup/health` (GET)
Health check endpoint.

## Integration with Other Features

To integrate with other API features:

1. Import the makeup router in your main API file:
```python
from src.api.makeup_routes import router as makeup_router
app.include_router(makeup_router)
```

2. Or create a unified API structure:
```python
# src/api/main.py
from fastapi import FastAPI
from src.api.makeup_routes import router as makeup_router
# ... other routers

app = FastAPI(title="TIVORA API")
app.include_router(makeup_router)
# ... include other routers
```

## Testing

Visit http://localhost:8000/docs for interactive API documentation (Swagger UI).

## Next Steps

1. Add authentication/authorization if needed
2. Add rate limiting
3. Add request validation
4. Add logging
5. Configure CORS for production
6. Add error handling improvements

