from steps.clean import Cleaner
from steps.train import Trainer
import logging
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main(status: Literal['sampling', 'all', 'export', 'validation', 'visualize_onnx']):
    if status != 'export' and status != 'visualize_onnx':
        logging.info("Starting the process")
        cleaner = Cleaner(status= 'count')
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
        path = trainer.export_model("yolov10n_development/20250106_160259/weights/best.pt")
        logging.info(f"Exporting Completed : {path}")
        
    elif status == 'visualize_onnx':
        trainer = Trainer()
        inp = input("Enter the model (1: 8n, 2: 10n, 3: 11n) -> ")
        sample_path = "sample/assignment_test_palm.jpeg"
        path_model = ""
        if inp == "1":
            path_model = "yolov8n_development/20250106_092214/weights/best.onnx"
        elif inp == "2":
            path_model = "yolov10n_development/20250106_160259/weights/best.onnx"
        elif inp == "3":
            path_model = "yolo11n_development/20250105_164322/weights/best.onnx"
        
        trainer.visualize(path_onnx= path_model, image_test= sample_path)
        # trainer.visualize(path_onnx= "yolov8n_development/20250106_092214/weights/best.onnx", image_test= "sample/assignment_test_palm.jpeg")
        
        
if __name__ == "__main__":
    inp = input("Enter the command (1: all, 2: sampling, 3: export, 4: vis_onnx) -> ")
    if inp == "1":
        main(status= 'all')
        
    elif inp == "2":
        main(status='sampling')
        
    elif inp == "3":
        main(status='export')
        
    elif inp == "4":
        main(status='visualize_onnx')