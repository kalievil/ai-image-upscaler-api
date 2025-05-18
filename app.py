"""
App entry point for Render deployment.
This file imports the FastAPI application from rapidapi_app.py and makes it available
for Gunicorn to serve.
"""

import os
import sys

# Import the app from rapidapi_app.py
from rapidapi_app import app

# This allows Gunicorn to find the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000))) 