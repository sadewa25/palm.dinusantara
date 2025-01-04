from steps.config import Configurations
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ultralytics import YOLO
from lightning.pytorch import Trainer as LTrainer
import mlflow
import glob
import os
import yaml
from ultralytics import settings

class Trainer(Configurations):
    def __init__(self):
        super().__init__()
        
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
        # experiment_name = self.config['model']['experiment']
        
        # Update a setting
        settings.update({"mlflow": True})

        # Reset settings to default values
        settings.reset()
        
        model_name = self.config['model']['name']
        epochs = self.config['train']['max_epochs']
        batch_size = self.config['train']['batch_size']
        image_size = self.config['preprocessing']['resize_img']
        

        # mlflow.set_experiment(experiment_name=experiment_name)
        
        # params = {
        #     "model": model_name,
        #     "epochs": epochs,
        #     "batch_size": batch_size,
        #     "img_size": image_size
        # }
        
        # # Log parameters
        # mlflow.log_params(params=params)
        
        # if mlflow.active_run():
        #     mlflow.end_run()
        
        # with mlflow.start_run() as run:

        # Load YOLO model
        model = YOLO(f'{model_name}.pt')  # Use pretrained YOLOv8n model

        # Train the model
        results = model.train(
            data= "data.yaml",
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size
        )
        
        return results
        
        # mlflow.set_tag("architecture", "YOLO")

        # # Log metrics
        # mlflow.log_metric("train_loss", results.results_dict['metrics/box_loss'])
        # mlflow.log_metric("val_loss", results.results_dict['metrics/val/box_loss'])
        # mlflow.log_metric("map50", results.results_dict['metrics/mAP_0.5'])
        # mlflow.log_metric("map50_95", results.results_dict['metrics/mAP_0.5:0.95'])

        # Save model weights
        # model_uri = f"runs:/{run.info.run_id}/model"
        # mlflow.register_model(model_uri, model_name)
        # model_path = "runs/train/weights/best.pt"
        # mlflow.log_artifact(model_path)
        
        # model.export(format='onnx', dynamic=True, simplify=True)
            
            # mlflow.end_run()
        