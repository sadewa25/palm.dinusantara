from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
import os
import cv2
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from ext import NotAllowedExtensions, setup_logger, DecodeImage, FileImageSize


logger = setup_logger()


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Food.Dinusantara", description="API for Indonesian Traditional Food", version="0.0.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/upload_classify")
@limiter.limit("5/minute")
async def upload_classify(request: Request, file: UploadFile = File(...)):
    # Validate file extension
    NotAllowedExtensions(file, logger=logger)
    
    try:
        contents = await file.read()
        # Image is valid
        DecodeImage(contents= contents, logger= logger)
        
        # Validate file size
        FileImageSize(contents= contents, logger= logger)
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        
        
        
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


