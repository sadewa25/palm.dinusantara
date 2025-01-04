from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class PalmDatasets(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.max_labels = 200
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Implement transform if it's not None
        if self.transform:
            image = self.transform(image)
        
        # Convert image to float32
        if isinstance(image, torch.Tensor):
            image = image.to(dtype=torch.float32)
            
        # Convert label to float32 if it's a tensor
        boxes = []
        with open(self.labels[idx], 'r') as file:
            for line in file:
                class_id, center_x, center_y, width, height = map(float, line.strip().split())
                boxes.append([class_id, center_x, center_y, width, height])
        
        # Convert labels to numpy array
        labels = np.array(boxes, dtype=np.float32)
        
        # Pad labels to ensure they have the same size
        if labels.shape[0] < self.max_labels:
            padding = np.zeros((self.max_labels - labels.shape[0], 5), dtype=np.float32)
            labels = np.vstack((labels, padding))
        elif labels.shape[0] > self.max_labels:
            labels = labels[:self.max_labels]
        
        # Convert labels to tensor
        label = torch.tensor(labels, dtype=torch.float32)
            
        return image, label