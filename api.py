from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
import os
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from ext import NotAllowedExtensions, setup_logger, DecodeImage, FileImageSize, PredictImg, FormatTempPath, ValidateImg
from ultralytics import YOLO
from datetime import datetime
from fastapi.staticfiles import StaticFiles


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

# Add at app initialization
app.mount("/static", StaticFiles(directory="static"), name="static")

# static_dir_count
static_count_dir = "static/counts"
static_classify_dir = "static/classify"




@app.post("/upload_count")
@limiter.limit("5/minute")
async def upload_count(request: Request, file: UploadFile = File(...)):
    
    # create folder when not exists
    os.makedirs(static_count_dir, exist_ok=True)
    
    try:
        contents = await file.read()
        
        ValidateImg(file= file, contents= contents, logger= logger)
        
        results, img, time_files = PredictImg(path= "yolov8n_development/20250106_092214/weights/best.onnx", contents= contents, static_count_dir= static_count_dir)
        
        # Draw boxes
        annotated_img = img.copy()
        
        for idx, box in enumerate(results.boxes.data.tolist(), start=1):
            x1, y1, x2, y2, _, _ = box
            cv2.rectangle(annotated_img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 0, 255), 1)
            
            cv2.putText(annotated_img,
                       f'{idx}',
                       (int(x1), int(y1-5)), # position slightly above box
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.2, # font scale
                       (0, 0, 255), # color (BGR)
                       1)


        # Resize to HD
        annotated_img_hd = cv2.resize(
            annotated_img, (1080, 1080), 
            interpolation=cv2.INTER_AREA
        )
        
        output_path = f"{static_count_dir}/detected_{time_files}.jpg"
        cv2.imwrite(output_path, annotated_img_hd)
        
        # delete original image from users
        os.remove(FormatTempPath(static_count_dir= static_count_dir, time_files= time_files))
        
        # change it with original image resize
        cv2.imwrite(FormatTempPath(static_count_dir= static_count_dir, time_files= time_files), img)

        return {
            "message": "Detection successful",
            "detections": len(results.boxes),
            "output_image": output_path
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


