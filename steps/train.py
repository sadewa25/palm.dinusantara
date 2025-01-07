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

class Trainer(Configurations):
    def __init__(self):
        super().__init__()
        self.batch_size = self.config['train']['batch_size']
        self.image_size = self.config['preprocessing']['resize_img']
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def yamlPreparation(self, status: str):
        root_path = self.config['data']['sampling'] if status == "sampling" else self.config['data']['root']
        
        train_path = os.path.abspath(os.path.join(root_path, 'train/images'))
        valid_path = os.path.abspath(os.path.join(root_path, 'valid/images'))
        test_path = os.path.abspath(os.path.join(root_path, 'test/images'))
        
        data_yaml = {
            'train': train_path,
            'val': valid_path,
            'test': test_path,
            'nc': self.config['data']['num_classes'],
            'names': [self.config['data']['names']]
        }
        
        with open('data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
            
        
    def train(self):
        model_name = self.config['model']['name']
        model_experiment = self.config['model']['experiment']
        epochs = self.config['train']['max_epochs']
        

        # Load YOLO model
        model = YOLO(f'{model_name}.pt')  # Use pretrained YOLOv8n model

        # Train the model
        model.train(
            data= "data.yaml",
            epochs=epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            device= get_device(),
            project=f"{model_name}_{model_experiment}",
            name=self.run_name,
        )
        
        return model
    
    def val_test(self, model):
        # Customize validation settings
        model.val(data="data.yaml", imgsz=self.image_size, batch=self.batch_size, device=get_device(), split='test', name= f"{self.run_name}_test")
    
    
    def export_model(self, path):
        model = YOLO(path)
        return model.export(
            format="onnx",
            dynamic=True,
            simplify=True,
        )
        
        
    def visualize(self, path_onnx: str, image_test: str):
        img_size = self.config['preprocessing']['resize_img']
        # Load model and run inference
        onnx_model = YOLO(path_onnx)
        img = cv2.imread(image_test)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (img_size, img_size))
        
        results = onnx_model.predict(img, 
            max_det=-1, 
            conf=0.25,        # Confidence threshold
            iou=0.45
        )[0]
        
        # Create figure and axes
        fig, ax = plt.subplots(1)
        
        # Load and display image using cv2
        ax.imshow(img)
        
        # Count total objects
        total_objects = len(results.boxes)
        plt.title(f'Total Objects Detected: {total_objects}', 
                pad=10, 
                fontsize=12, 
                fontweight='bold')
        
        # Plot each detection
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            
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
        
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/model/{current_time}.png"
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return results, save_path
        
        
        
        