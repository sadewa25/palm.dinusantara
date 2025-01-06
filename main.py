import numpy as np
import cv2
from steps.clean import Cleaner
from steps.train import Trainer
import logging
from typing import Literal


# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')


def main(status: Literal['sampling', 'all', 'export', 'validation', 'visualize_onnx']):
    if status != 'export' and status != 'visualize_onnx':
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
        logging.info("Training the model")
        modelTrain = trainer.train()
        logging.info("Validation the model")
        trainer.val_test(model= modelTrain)
        logging.info("Process Completed")
        
        
    elif status == 'export':
        trainer = Trainer()
        logging.info("Exporting Progress")
        path = trainer.export_model("yolo11n_development/20250105_164322/weights/best.pt")
        logging.info(f"Exporting Completed : {path}")
        
    elif status == 'visualize_onnx':
        trainer = Trainer()
        trainer.visualize(path_onnx= "yolo11n_development/20250105_164322/weights/best.onnx", image_test= "data/test/images/DJI_0098_0_JPG.rf.272d7237869e77ea358dd11bb3fa9e95.jpg")
        
    
    
if __name__ == "__main__":
    # main(status= 'all')
    # main(status='sampling')
    # main(status='export')
    main(status='visualize_onnx')
    # main(status='sam')