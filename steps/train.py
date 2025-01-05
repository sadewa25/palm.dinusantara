from steps.config import Configurations
from ultralytics import YOLO
import os
import yaml
from ultralytics import settings
from utils import get_device
from datetime import datetime

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
        
        