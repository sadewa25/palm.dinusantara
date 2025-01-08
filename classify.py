from steps.clean import Cleaner
from steps.train import Trainer
import logging
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main(status: Literal['sampling', 'all', 'export', 'validation', 'visualize_onnx']):
    if status != 'export' and status != 'visualize_onnx':
        logging.info("Starting the process")
        cleaner = Cleaner(status= 'classify')
        logging.info("Rename Directory")
        cleaner.rename_directory()
        
        if status == 'sampling':
            logging.info("Sampling Images")
            cleaner.sampling_images(5)
            
        trainer = Trainer('classify')
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
        path = trainer.export_model("yolov9t_development/20250107_113955/weights/best.pt")
        logging.info(f"Exporting Completed : {path}")
        

if __name__ == "__main__":
    inp = input("1: all, 2: sampling, 3: export, 4: validation, 5: visualize_onnx -> ")
    
    if inp == "1":
        main("all")
    
    elif inp == "2":
        main("sampling")