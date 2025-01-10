from steps.config import Configurations
from ultralytics import YOLO
import os
import yaml
from ultralytics import settings
from utils import get_device
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from typing import Literal
import numpy as np

class Trainer(Configurations):
    def __init__(self, status: Literal['count', 'classify', 'resume'] = 'count'):
        super().__init__()
        self.status = 'data_count' if status == 'count' else 'data_classify'
        self.batch_size = self.config['train']['batch_size']
        self.image_size = self.config['preprocessing']['resize_img']
        self.ext_machine = '_classify' if self.status == "data_classify" else ''
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{self.config['train']['max_epochs']}" + self.ext_machine
        
    def yamlPreparation(self, status: Literal['sampling', 'all'] = 'all'):
        root_path = self.config[self.status]['sampling'] if status == 'sampling' else self.config[self.status]['root']
        
        train_path = os.path.abspath(os.path.join(root_path, 'train/images'))
        valid_path = os.path.abspath(os.path.join(root_path, 'valid/images'))
        test_path = os.path.abspath(os.path.join(root_path, 'test/images'))
        
        data_yaml = {
            'train': train_path,
            'val': valid_path,
            'test': test_path,
            'nc': self.config[self.status]['num_classes'],
            'names': [self.config[self.status]['names']]
        }
        
        with open(self.config[self.status]['yaml'], 'w') as f:
            yaml.dump(data_yaml, f)
            
        
    def train(self, status_train: Literal['start', 'resume'] = 'start', path: str = ''):
        model_name = self.config['model']['name']
        model_experiment = self.config['model']['experiment']
        epochs = self.config['train']['max_epochs']
        
        # Load YOLO model yolo11m.pt
        model = YOLO(f'{model_name}.pt' if status_train == 'start' else path, task= 'detect')  # Use pretrained YOLOv8n model
        is_resume = status_train == 'resume'

        # Train the model
        model.train(
            data= self.config[self.status]['yaml'],
            epochs=epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            device= get_device(),
            project=f"{model_name}_{model_experiment}",
            name=self.run_name,
            optimizer='Adam',
            resume= is_resume
        )
        
        return model
    
    def val_test(self, model):
        # Customize validation settings
        model.val(data=self.config[self.status]['yaml'], imgsz=self.image_size, batch=self.batch_size, device=get_device(), split='test', name= f"{self.run_name}_test")
    
    
    def export_model(self, path):
        model = YOLO(path)
        return model.export(
            format="onnx",
            dynamic=True,
            simplify=True,
        )
        
        
    def visualize(self, path_onnx: str, image_test: str):
        img_size = self.config['preprocessing']['resize_img']
        
        # Get the base path
        base_model = os.path.dirname(path_onnx).split('/')[0].split('_')[0]
        
        # Load model and run inference
        onnx_model = YOLO(path_onnx)
        img = cv2.imread(image_test)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (img_size, img_size))
        
        results = onnx_model.predict(img, 
            max_det=-1, 
            conf=0.25,        # Confidence threshold
            iou=0.45,
            task='detect'
        )[0]
        
        # Create figure and axes
        _, ax = plt.subplots(1)
        
        # Load and display image using cv2
        ax.imshow(img)
        
        # Count total objects
        total_objects = len(results.boxes)
        plt.title(f'Total Objects Detected: {total_objects}', 
                pad=10, 
                fontsize=12, 
                fontweight='bold')
        

        # Define color ranges for classification
        color_ranges = {
            'red': ([0, 0, 100], [80, 80, 255]),        # Lower and Upper range for red
            'yellow': ([0, 100, 100], [100, 255, 255]), # Yellow
            'green': ([0, 100, 0], [100, 255, 100])     # Green
        }
        
        counts = {color: 0 for color in color_ranges}
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_ext = f"{self.config['output']['root']}/{self.config['output']['model']}/{base_model}_{current_time}{self.ext_machine}"
        
        if not os.path.exists(name_ext):
            os.makedirs(name_ext)
        
        # Plot each detection
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            
            # Get region of interest within rectangle
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            
            # Convert to HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            # Define yellow range
            label = 'unknown'
            for color, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
                if cv2.countNonZero(mask) > 0:  # Check if color exists
                    label = color
                    counts[color] += 1
                    break
            
            back_img = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{name_ext}/{label}_{counts[label]}.jpg", back_img)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), 
                x2-x1, 
                y2-y1, 
                linewidth=2, 
                edgecolor='r', 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            # label = f"{conf:.2f}"
            # plt.text(x1, y1, label, color='white', bbox=dict(facecolor='red', alpha=0.5))
        
        save_path = f"{name_ext}.png"
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return results, save_path
        
        
        
        