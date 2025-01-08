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
            

if __name__ == "__main__":
    main("sampling")