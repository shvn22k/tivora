"""
Makeup API Server - Uses existing src module
Port: 8001
"""
import sys
sys.path.insert(0, '../../')

from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
