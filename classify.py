from steps.clean import Cleaner
from steps.train import Trainer
import logging
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main(status: Literal['sampling', 'all', 'export', 'validation', 'visualize_onnx', 'resume']):
    trainer = Trainer("classify")

    if status != 'export' and status != 'visualize_onnx' and status != 'resume':
        logging.info("Starting the process")
        cleaner = Cleaner(status= 'classify')
        logging.info("Rename Directory")
        cleaner.rename_directory()
        
        if status == 'sampling':
            logging.info("Sampling Images")
            cleaner.sampling_images(5)
            
        logging.info("Create data.yaml")
        trainer.yamlPreparation(status= status)
        logging.info("Training the model")
        modelTrain = trainer.train()
        logging.info("Validation the model")
        trainer.val_test(model= modelTrain)
        logging.info("Process Completed")
        
    elif status == 'export':
        logging.info("Exporting Progress")
        path = trainer.export_model("yolo11n_development/20250109_045532_128_classify/weights/best.pt")
        logging.info(f"Exporting Completed : {path}")
        
    elif status == 'visualize_onnx':
        inp = input("Enter the model (1: 8n, 2: 9t, 3: 10n, 4: 11n, 5: 11s) -> ")
        sample_path = "sample/assignment_test_apple.jpeg"
        if inp == "1":
            pass
        if inp == "2":
            pass
        elif inp == "3":
            pass
        elif inp == "4":
            path_model = "yolo11n_development/20250109_045532_128_classify/weights/best.onnx"
        elif inp == "5":
            pass
        trainer.visualize(path_onnx= path_model, image_test= sample_path)

    
    elif status == 'resume':
        logging.info("Create data.yaml")
        trainer.yamlPreparation()
        logging.info("Training the model")
        modelTrain = trainer.train(status_train= 'resume', path= "yolo11n_development/20250109_045532_128_classify/weights/last.pt")
        logging.info("Validation the model")
        trainer.val_test(model= modelTrain)
        logging.info("Process Completed")
        
        

if __name__ == "__main__":
    inp = input("Enter the command (1: all, 2: sampling, 3: export, 4: vis_onnx, 5: resume) -> ")
    
    if inp == "1":
        main("all")
    
    elif inp == "2":
        main("sampling")
        
    elif inp == "3":
        main("export")
        
    elif inp == "4":
        main("visualize_onnx")

    elif inp == "5":
        main("resume")