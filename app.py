import os
import sys

# Set environment variable to development for testing
os.environ[\
ENVIRONMENT\] = \development\

# Import the app from rapidapi_app.py
from rapidapi_app import app

# Override the middleware to allow direct access
@app.middleware(\
http\)
async def override_rapidapi_headers(request, call_next):
    # Always allow the request through
    return await call_next(request)

# This allows Uvicorn to find the app
if __name__ == \
__main__\:
    import uvicorn
    uvicorn.run(\
app:app\, host=\0.0.0.0\, port=int(os.environ.get(\PORT\, 8000)))
