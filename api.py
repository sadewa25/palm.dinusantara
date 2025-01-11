from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
import os
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from ext import NotAllowedExtensions, setup_logger, DecodeImage, FileImageSize
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
        onnx_model = YOLO("yolo11n_development/20250109_045532_128_classify/weights/best.onnx")
        results = onnx_model.predict(img, 
            max_det=-1, 
            conf=0.25,        # Confidence threshold
            iou=0.45,
            task='classify'
        )[0]
        print(results)
        
    except Exception as e:
        print("Error uploading file: ", str(e))



@app.post("/upload_count")
@limiter.limit("5/minute")
async def upload_count(request: Request, file: UploadFile = File(...)):
    
    # create folder when not exists
    os.makedirs(static_count_dir, exist_ok=True)
    
    # Validate file extension
    NotAllowedExtensions(file, logger=logger)
    
    try:
        contents = await file.read()
        # Image is valid
        DecodeImage(contents= contents, logger= logger)
        
        # Validate file size
        FileImageSize(contents= contents, logger= logger)
        # Save uploaded file temporarily
        time_files = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = f"{static_count_dir}/temp_{time_files}.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Load and process image
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        
        onnx_model = YOLO("yolov8n_development/20250106_092214/weights/best.onnx")
        results = onnx_model.predict(img, 
            max_det=-1, 
            conf=0.25,        # Confidence threshold
            iou=0.45,
            task='detect'
        )[0]
        
        # Draw boxes
        annotated_img = img.copy()
        
        for idx, box in enumerate(results.boxes.data.tolist(), start=1):
            x1, y1, x2, y2, conf, cls = box
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
        annotated_img_hd = cv2.resize(annotated_img, (1080, 1080), 
                                    interpolation=cv2.INTER_AREA)
        
        # Enhance image quality
        # annotated_img_hd = cv2.detailEnhance(annotated_img_hd, sigma_s=10, sigma_r=0.15)
        output_path = f"{static_count_dir}/detected_{time_files}.jpg"
        cv2.imwrite(output_path, annotated_img_hd)
        
        # delete original image from users
        os.remove(temp_path)
        
        # change it with original image resize
        cv2.imwrite(temp_path, img)

        return {
            "message": "Detection successful",
            "detections": len(results.boxes),
            "output_image": output_path
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


