import numpy as np
import cv2
from steps.clean import Cleaner
from steps.train import Trainer
import logging
from typing import Literal
import yaml
import mlflow


# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')


def main(status: Literal['sampling', 'all']):
    config = None
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
        
    logging.info("Starting the process")
    cleaner = Cleaner()
    logging.info("Rename Directory")
    cleaner.rename_directory()
    
    if status == 'sampling':
        logging.info("Sampling Images")
        cleaner.sampling_images(5)
    
    
    # cleaner.download_datasets()
    # unzip the downloaded datasets
    # rename the folder to 'data'
    
    # check label images is exists
    # cleaner.isLabelExists('train')
    # cleaner.isLabelExists('valid')
    
    # check order of images and labels
    # cleaner.isOrderSame('train')
    # cleaner.isOrderSame('valid')
    
    # train_loader, val_loader = cleaner.prepare_datasets()
    # generate visualization of the data
    # cleaner.visualize_data(train_loader)
    
    trainer = Trainer()
    logging.info("Create data.yaml")
    trainer.yamlPreparation(status= status)
    
    trainer.train()
    
    
    
    
    
if __name__ == "__main__":
    main(status='sampling')