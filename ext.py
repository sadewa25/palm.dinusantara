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
    
    
def preprocess_image(image_path: str):
    """Preprocess the image for ONNX model inference."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Resize to the model's input size
    img = img.astype(np.float32)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_output(output: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """Postprocess the ONNX model output to filter predictions."""
    predictions = []
    for pred in output:
        boxes, scores, labels = pred[:, :4], pred[:, 4], pred[:, 5].astype(int)
        for box, score, label in zip(boxes, scores, labels):
            if score >= conf_threshold:
                predictions.append({"box": box, "score": score, "label": label})
    return predictions

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
    
    for idx, box in enumerate(results.boxes.data.tolist(), start=1):
            x1, y1, x2, y2, _, _ = box
            cv2.rectangle(img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 0, 255), 1)
            
            cv2.putText(img,
                       f'{idx}',
                       (int(x1), int(y1-5)), # position slightly above box
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.2, # font scale
                       (0, 0, 255), # color (BGR)
                       1)
    
    return results, img, time_files