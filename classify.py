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
        logging.info("Remove Unused Files")
        cleaner.unused_datasets()
        
        if status == 'sampling':
            logging.info("Sampling Images")
            cleaner.sampling_images(5)
            

if __name__ == "__main__":
    main("sampling")