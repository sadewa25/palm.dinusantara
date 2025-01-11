import os
from fastapi import HTTPException
import numpy as np
import cv2
import logging
from datetime import datetime

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 5_000_000  # 5MB

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create logger
    logger = logging.getLogger('food_dinusantara')
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create file handler
    file_handler = logging.FileHandler(
        f'logs/app_{datetime.now().strftime("%Y-%m-%d")}.log'
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    return logger
    
def NotAllowedExtensions(file, logger):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.error("File type not allowed")
        raise HTTPException(status_code=400, detail="File type not allowed")

def DecodeImage(contents, logger: any):
    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    
    # Decode image
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("File type not allowed")
            raise HTTPException(status_code=400, detail="File type not allowed")
            
    except Exception:
        logger.error("Error decoding image")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

def FileImageSize(contents, logger: any):
    if len(contents) > MAX_FILE_SIZE:
        logger.error("Error Max File Image Size")
        raise HTTPException(status_code=400, detail="File too large")