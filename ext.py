import os
from fastapi import HTTPException
import numpy as np
import cv2
import logging
from datetime import datetime
from ultralytics import YOLO
from fastapi import UploadFile


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 30_000_000  # 5MB

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

def FormatTempPath(static_count_dir: str, time_files: str):
    return f"{static_count_dir}/temp_{time_files}.jpg"

def ValidateImg(file: UploadFile, contents: bytes, logger: any):
    NotAllowedExtensions(file, logger= logger)
    
    # Image is valid
    DecodeImage(contents= contents, logger= logger)
    
    # Validate file size
    FileImageSize(contents= contents, logger= logger)
    

def PredictImg(path: str, contents: bytes, static_count_dir: str):
    
    # Save uploaded file temporarily
    time_files = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_path = FormatTempPath(static_count_dir= static_count_dir, time_files= time_files)
    with open(temp_path, "wb") as f:
        f.write(contents)

    # Load and process image
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    
    onnx_model = YOLO(path)
    results = onnx_model.predict(img, 
        max_det=-1, 
        conf=0.25,        # Confidence threshold
        iou=0.45,
        task='detect'
    )[0]
    
    return results, img, time_files