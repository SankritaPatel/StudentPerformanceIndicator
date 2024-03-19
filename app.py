import sys
from src.StudentPerformanceIndicator.logger import logging
from src.StudentPerformanceIndicator.exception import CustomException
from src.StudentPerformanceIndicator.components.data_ingestion import DataIngestion
from src.StudentPerformanceIndicator.components.data_transformation import DataTransformation
from src.StudentPerformanceIndicator.components.model_trainer import ModelTrainer

if __name__=="__main__":
    logging.info("The execution has started...")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        model_trainer = ModelTrainer()
        r2Score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(r2Score)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
