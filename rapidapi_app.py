import os
import sys
import json
import time
import uuid
import shutil
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Optional, List
import torch
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import warnings
import uvicorn
from io import BytesIO
import logging
import base64
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rapidapi_app")

# Create directories
os.makedirs("models/realesrgan", exist_ok=True)
os.makedirs("storage", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Global flag to track RealESRGAN availability
REALESRGAN_AVAILABLE = False

# Try importing RealESRGAN components
try:
    logger.info("Attempting to import RealESRGAN...")
    # Import directly from the realesrgan packages
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    logger.info("Successfully imported RealESRGAN modules!")
    REALESRGAN_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import RealESRGAN: {str(e)}")
    logger.info("Falling back to PIL upscaler")
    REALESRGAN_AVAILABLE = False

# Define model and file format enums
class UpscalerModel(str, Enum):
    realesrgan_x4 = "realesrgan-x4"
    realesrgan_anime = "realesrgan-anime"
    standard = "standard"
    sharp = "sharp"
    smooth = "smooth"

class OutputFormat(str, Enum):
    png = "png"
    jpg = "jpg"
    webp = "webp"

# Initialize FastAPI app with RapidAPI settings
app = FastAPI(
    title="AI Image Upscaler API",
    description="High-quality AI-powered image upscaler using RealESRGAN and other enhancement methods",
    version="1.0.0",
    docs_url="/documentation",
    redoc_url=None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model storage
realesrgan_models = {}

# RapidAPI header verification middleware
@app.middleware("http")
async def verify_rapidapi_headers(request: Request, call_next):
    # For local development, bypass header check
    if os.environ.get("ENVIRONMENT") == "development":
        return await call_next(request)
    
    # Check for the RapidAPI headers
    rapidapi_proxy_secret = request.headers.get("X-RapidAPI-Proxy-Secret")
    rapidapi_host = request.headers.get("X-RapidAPI-Host")
    
    # Skip header check for documentation endpoints
    if request.url.path in ["/documentation", "/openapi.json"]:
        return await call_next(request)
    
    # Validate headers for all other endpoints
    if not rapidapi_proxy_secret or not rapidapi_host:
        return JSONResponse(
            status_code=403,
            content={"detail": "Missing RapidAPI headers"}
        )
        
    # Continue with the request
    return await call_next(request)

@app.on_event("startup")
async def startup_event():
    global REALESRGAN_AVAILABLE, realesrgan_models
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize RealESRGAN models if available
    if REALESRGAN_AVAILABLE:
        try:
            logger.info("Loading RealESRGAN models...")
            
            # Check if model files exist
            model_x4_path = "models/realesrgan/RealESRGAN_x4plus.pth"
            model_anime_path = "models/realesrgan/RealESRGAN_x4plus_anime_6B.pth"
            
            missing_models = []
            if not os.path.exists(model_x4_path):
                missing_models.append(model_x4_path)
            if not os.path.exists(model_anime_path):
                missing_models.append(model_anime_path)
                
            if missing_models:
                logger.warning(f"Missing model files: {missing_models}")
                logger.warning("Please download the models")
                # Try to download models
                logger.info("Attempting to download missing models...")
                try:
                    os.system("python download_models.py")
                    logger.info("Models downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download models: {str(e)}")
                    REALESRGAN_AVAILABLE = False
            
            # Check again if model files exist after download attempt
            if os.path.exists(model_x4_path) and os.path.exists(model_anime_path):
                logger.info("All model files found, proceeding with initialization...")
                
                # Initialize RealESRGAN x4 model
                # Using the correct architecture for the model
                model_x4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                realesrgan_models['x4'] = RealESRGANer(
                    scale=4,
                    model_path=model_x4_path,
                    model=model_x4,
                    tile=400,  # Use tile mode for large images
                    tile_pad=10,
                    pre_pad=0,
                    half=False if device.type == 'cpu' else True,  # Use half precision on GPU
                    device=device
                )
                
                # Initialize RealESRGAN anime model
                model_anime = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                realesrgan_models['anime'] = RealESRGANer(
                    scale=4,
                    model_path=model_anime_path,
                    model=model_anime,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False if device.type == 'cpu' else True,
                    device=device
                )
                
                logger.info("✅ RealESRGAN models loaded successfully!")
            else:
                logger.error("❌ Model files still missing after download attempt")
                REALESRGAN_AVAILABLE = False
        except Exception as e:
            logger.error(f"❌ Error loading RealESRGAN models: {str(e)}")
            logger.info("Falling back to PIL-based upscaling")
            REALESRGAN_AVAILABLE = False
    
    logger.info("✅ AI Image Upscaler API is ready!")

def enhance_image_pil(img, scale, denoise_level=0, model_type="standard"):
    """Enhance image using PIL with different styles"""
    # Get original size
    width, height = img.size
    
    # Resize with high quality
    img = img.resize((width * scale, height * scale), Image.LANCZOS)
    
    # Apply different enhancement styles
    if model_type == "sharp":
        # High sharpness, high contrast
        img = img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Very sharp
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # Higher contrast
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.1)  # Slightly more vibrant
    
    elif model_type == "smooth":
        # Smoothed, softer image
        img = img.filter(ImageFilter.SMOOTH_MORE)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(0.8)  # Less sharp
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.9)  # Slightly lower contrast
    
    else:  # standard
        # Apply denoising if requested
        if denoise_level > 0:
            # Apply stronger smoothing for higher denoise levels
            for _ in range(denoise_level):
                img = img.filter(ImageFilter.SMOOTH_MORE)
        
        # Standard enhancements
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)  # Sharpen by 50%
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Slight contrast boost
    
    return img

def process_image(image_data, model, denoise_level=0, output_format="png", quality=95):
    """Process an image with the selected model and return the result"""
    try:
        # Open image from bytes
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed (for PNG with transparency)
        if img.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGBA', img.size, (255, 255, 255))
            # Composite the image with the background
            img = Image.alpha_composite(background, img).convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Process image based on selected model
        if model.startswith("realesrgan") and REALESRGAN_AVAILABLE:
            # Convert PIL image to OpenCV format (numpy array)
            cv_img = np.array(img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            
            # Choose the appropriate model
            model_key = 'anime' if model == 'realesrgan-anime' else 'x4'
            
            # Process with RealESRGAN
            output, _ = realesrgan_models[model_key].enhance(cv_img, outscale=4)
            
            # Apply additional processing based on denoise level
            if denoise_level > 0:
                # Apply bilateral filter for denoising while preserving edges
                sigma_color = denoise_level * 10  # Scale denoise level
                sigma_space = denoise_level * 5
                output = cv2.bilateralFilter(output, 9, sigma_color, sigma_space)
            
            # Convert back to PIL for saving
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convert BGR back to RGB
            result_img = Image.fromarray(output)
        else:
            # Use PIL-based upscaling as fallback
            scale = 4 if model.startswith("realesrgan") else 2
            model_type = "standard"
            
            if model == "sharp":
                model_type = "sharp"
            elif model == "smooth":
                model_type = "smooth"
                
            result_img = enhance_image_pil(img, scale, denoise_level, model_type)
        
        # Save to bytes with the specified format
        output_buffer = BytesIO()
        
        if output_format == "jpg":
            result_img.save(output_buffer, format="JPEG", quality=quality)
        elif output_format == "webp":
            result_img.save(output_buffer, format="WEBP", quality=quality)
        else:  # Default to PNG
            result_img.save(output_buffer, format="PNG")
        
        output_buffer.seek(0)
        return output_buffer.getvalue(), result_img.width, result_img.height
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "AI Image Upscaler API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/documentation"
    }

@app.get("/info")
async def get_info():
    """Get information about the API and available models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        "name": "AI Image Upscaler API",
        "version": "1.0.0",
        "enhanced_models_available": REALESRGAN_AVAILABLE,
        "upscaler_models": [model.value for model in UpscalerModel],
        "output_formats": [fmt.value for fmt in OutputFormat],
        "device": {
            "type": device.type,
            "index": device.index,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        },
        "max_image_size": "No specific limit, but larger images may take longer to process"
    }

@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    model: UpscalerModel = Form(UpscalerModel.realesrgan_x4),
    denoise_level: int = Form(0),
    output_format: OutputFormat = Form(OutputFormat.png),
    quality: int = Form(95)
):
    """
    Upscale an image using AI models
    
    - **file**: The image file to upscale
    - **model**: The upscaling model to use
    - **denoise_level**: Level of denoising to apply (0-3)
    - **output_format**: Output format (png, jpg, webp)
    - **quality**: Quality for jpg/webp formats (1-100)
    """
    # Validate parameters
    if denoise_level < 0 or denoise_level > 3:
        raise HTTPException(status_code=400, detail="Denoise level must be between 0 and 3")
    
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
    
    try:
        # Read the file
        start_time = time.time()
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process the image
        image_data, width, height = process_image(
            contents,
            model.value,
            denoise_level,
            output_format.value,
            quality
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate unique file identifier
        file_id = str(uuid.uuid4())
        
        # Save processed image to storage
        output_path = os.path.join("storage", f"{file_id}.{output_format.value}")
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        # Convert image to base64 for direct response
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "file_id": file_id,
                "original_filename": file.filename,
                "width": width,
                "height": height,
                "model_used": model.value,
                "output_format": output_format.value,
                "processing_time": round(processing_time, 2),
                "image_base64": encoded_image
            }
        }
    except Exception as e:
        logger.error(f"Error during upscale: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/url-upscale")
async def upscale_from_url(
    image_url: str = Form(...),
    model: UpscalerModel = Form(UpscalerModel.realesrgan_x4),
    denoise_level: int = Form(0),
    output_format: OutputFormat = Form(OutputFormat.png),
    quality: int = Form(95)
):
    """
    Upscale an image from a URL using AI models
    
    - **image_url**: The URL of the image to upscale
    - **model**: The upscaling model to use
    - **denoise_level**: Level of denoising to apply (0-3)
    - **output_format**: Output format (png, jpg, webp)
    - **quality**: Quality for jpg/webp formats (1-100)
    """
    # Validate parameters
    if denoise_level < 0 or denoise_level > 3:
        raise HTTPException(status_code=400, detail="Denoise level must be between 0 and 3")
    
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
    
    try:
        # Validate URL
        if not image_url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")
        
        # Download the image
        start_time = time.time()
        try:
            response = requests.get(image_url, timeout=30)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: HTTP {response.status_code}")
            
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"URL does not point to an image. Content-Type: {content_type}")
            
            contents = response.content
            if len(contents) == 0:
                raise HTTPException(status_code=400, detail="Downloaded image is empty")
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")
        
        # Process the image
        image_data, width, height = process_image(
            contents,
            model.value,
            denoise_level,
            output_format.value,
            quality
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate unique file identifier
        file_id = str(uuid.uuid4())
        
        # Save processed image to storage
        output_path = os.path.join("storage", f"{file_id}.{output_format.value}")
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        # Convert image to base64 for direct response
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "file_id": file_id,
                "source_url": image_url,
                "width": width,
                "height": height,
                "model_used": model.value,
                "output_format": output_format.value,
                "processing_time": round(processing_time, 2),
                "image_base64": encoded_image
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during URL upscale: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-upscale")
async def batch_upscale_images(
    files: List[UploadFile] = File(...),
    model: UpscalerModel = Form(UpscalerModel.realesrgan_x4),
    denoise_level: int = Form(0),
    output_format: OutputFormat = Form(OutputFormat.png),
    quality: int = Form(95)
):
    """
    Batch upscale multiple images using AI models
    
    - **files**: List of image files to upscale (max 5)
    - **model**: The upscaling model to use
    - **denoise_level**: Level of denoising to apply (0-3)
    - **output_format**: Output format (png, jpg, webp)
    - **quality**: Quality for jpg/webp formats (1-100)
    """
    # Limit batch size
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed for batch processing")
    
    # Validate parameters
    if denoise_level < 0 or denoise_level > 3:
        raise HTTPException(status_code=400, detail="Denoise level must be between 0 and 3")
    
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")
    
    results = []
    errors = []
    
    for file in files:
        try:
            # Read the file
            start_time = time.time()
            contents = await file.read()
            
            if len(contents) == 0:
                errors.append({
                    "filename": file.filename,
                    "error": "Empty file"
                })
                continue
            
            # Process the image
            image_data, width, height = process_image(
                contents,
                model.value,
                denoise_level,
                output_format.value,
                quality
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate unique file identifier
            file_id = str(uuid.uuid4())
            
            # Save processed image to storage
            output_path = os.path.join("storage", f"{file_id}.{output_format.value}")
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            # Convert image to base64 for direct response
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            
            results.append({
                "file_id": file_id,
                "original_filename": file.filename,
                "width": width,
                "height": height,
                "model_used": model.value,
                "output_format": output_format.value,
                "processing_time": round(processing_time, 2),
                "image_base64": encoded_image
            })
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    if not results and errors:
        # If all files failed
        raise HTTPException(status_code=500, detail={"message": "All files failed to process", "errors": errors})
    
    return {
        "status": "success",
        "processed": len(results),
        "failed": len(errors),
        "data": results,
        "errors": errors if errors else None
    }

if __name__ == "__main__":
    # For local development, set environment
    os.environ["ENVIRONMENT"] = "development"
    
    # Run the app
    uvicorn.run("rapidapi_app:app", host="0.0.0.0", port=8000, reload=True) 