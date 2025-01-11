import torch
import os
from typing import Literal
import shutil

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def open_txt(filePath: str):
    temp = []
    with open(filePath, 'r') as file:
        for line in file:
            temp.append(line.strip())
    
    return temp

def create_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def move_files(paths: list, root_path: str, dest: str, status: Literal['image', 'label']):
    ext = ".png" if status == 'image' else ".csv"
    
    if not os.path.exists(dest):
        os.mkdir(dest)
        
    for i in paths:
        img = f"{root_path}/{i}{ext}"
        if os.path.exists(img):
            shutil.move(img, dest)
            
