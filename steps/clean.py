from steps.config import Configurations
import glob
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import Literal
from datasets.palm_datasets import PalmDatasets
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from roboflow import Roboflow
import shutil


class Cleaner(Configurations):
    def __init__(self):
        super().__init__()
        base_path = self.config['data']['root']
        train_path = self.config['data']['train_path']
        valid_path = self.config['data']['valid_path']
        
        self.img_train_path = f"{base_path}/{train_path}/images"
        self.label_train_path = f"{base_path}/{train_path}/labels"
        self.img_valid_path = f"{base_path}/{valid_path}/images"
        self.label_valid_path = f"{base_path}/{valid_path}/labels"
        
        self.img_train = glob.glob(f"{self.img_train_path}/*.jpg")
        self.label_train = glob.glob(f"{self.label_train_path}/*.txt")
        self.img_train.sort()
        self.label_train.sort()
        
        self.img_val = glob.glob(f"{self.img_valid_path}/*.jpg")
        self.label_val = glob.glob(f"{self.label_valid_path}/*.txt")
        self.img_val.sort()
        self.label_val.sort()
        
    def move_files(self, data: list[str], sampling_dir: str):
        for path in data:
            # Determine the relative path of the image
            relative_path = os.path.relpath(path, self.config['data']['root'])
            # Determine the destination path in the sampling directory
            dest_path = os.path.join(sampling_dir, relative_path)
            # Create the necessary subdirectories in the sampling directory
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # Copy the image to the sampling directory
            shutil.copy(path, dest_path)
        
    def sampling_images(self, number_sampling: int):
        # Create the sampling directory if it doesn't exist
        sampling_dir = self.config['data']['sampling']
        os.makedirs(sampling_dir, exist_ok=True)
        
        self.move_files(data= self.img_train[:number_sampling], sampling_dir= sampling_dir)
        self.move_files(data= self.label_train[:number_sampling], sampling_dir= sampling_dir)
        self.move_files(data= self.img_val[:number_sampling], sampling_dir= sampling_dir)
        self.move_files(data= self.label_val[:number_sampling], sampling_dir= sampling_dir)
        
        
    def rename_directory(self):
        old_name = 'Palm Tree Label 200m Splitted.v7i.yolov11'
        new_name = self.config['data']['root']
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
        
    def download_datasets(self):
        rf = Roboflow(api_key="LjjjZ43esu69CcKuFaCz")
        project = rf.workspace("cryospace-yylkk").project("palm-tree-label-200m-splitted")
        version = project.version(7)
        dataset = version.download("yolov11")
        return dataset
    
    def isLabelExists(self, status: Literal['train', 'valid']):
        img_status = self.img_train if status == 'train' else self.img_val
        label_status = self.label_train if status == 'train' else self.label_val
        
        label_train = [os.path.basename(label) for label in label_status]
        count_false = 0
        for i in tqdm(img_status, desc="Checking labels"):
            file_label = i.split("/")[-1].replace(".jpg", ".txt")
            if file_label not in label_train:
                count_false += 1
        
        if count_false == 0:
            print(f"All images {status} have labels")
        else:
            print(f"Total False: {count_false}")
            
    def isOrderSame(self, status: Literal['train', 'valid']):
        order_wrong = []
        img_status = self.img_train if status == 'train' else self.img_val
        label_status = self.label_train if status == 'train' else self.label_val
        
        for i in tqdm(range(len(img_status)), desc="Checking order"):
            file_label = img_status[i].split("/")[-1].replace(".jpg", ".txt")
            if file_label != os.path.basename(label_status[i]):
                order_wrong.append(file_label)
                
        if len(order_wrong) == 0:
            print(f"All orders data {status} are correct")
        else:
            print(f"Orders wrong: {order_wrong} / {len(order_wrong)}")
    
    def visualize_data(self, loader: DataLoader):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Training Images Sample')
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        train_iter = next(iter(loader))
        image, label = train_iter
        images = image[:3]
        
        for i in range(3):
            image = images[i].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            
            height, width, _ = image.shape
            axes[i].imshow(image)
            axes[i].axis('off')
            
            # Draw bounding boxes
            for box in label[i]:
                class_id, x_center, y_center, w, h = box
                
                # Convert normalized coordinates to pixel coordinates
                x_center = int(x_center * width)
                y_center = int(y_center * height)
                w = int(w * width)
                h = int(h * height)
                
                # Calculate top-left corner from center coordinates
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                axes[i].add_patch(rect)

        plt.tight_layout()
        path_output_clean = f"{self.config['output']['root']}/{self.config['output']['clean']}"
        plt.savefig(f"{path_output_clean}/training_samples_{current_time}.png")
        plt.close()
    
    def prepare_datasets(self):
        resize_img = self.config['preprocessing']['resize_img']
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_img, resize_img)),
            transforms.ToTensor()
        ])
        
        self.img_train = self.img_train[:10]
        self.label_train = self.label_train[:10]
        
        self.img_val = self.img_val[:10]
        self.label_val = self.label_val[:10]
        
        train_ds = PalmDatasets(self.img_train, self.label_train, transform=transform)
        val_ds = PalmDatasets(self.img_val, self.label_val, transform=transform)
        
        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=self.config['train']['batch_size'], shuffle=True, num_workers=self.config['train']['num_workers'], persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=self.config['train']['batch_size'], num_workers=self.config['train']['num_workers'], persistent_workers=True)
        
        return train_loader, val_loader
        
        
        
        