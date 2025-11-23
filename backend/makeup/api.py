import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    print("Starting makeup API...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
