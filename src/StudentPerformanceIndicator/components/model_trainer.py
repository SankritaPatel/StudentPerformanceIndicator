import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import(
    GradientBoostingRegressor, 
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.StudentPerformanceIndicator.exception import CustomException
from src.StudentPerformanceIndicator.logger import logging
from src.StudentPerformanceIndicator.utils import save_object, evaluate_models
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }   
            }
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models, params)
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            #To  get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print("THis is the best model:", best_model_name)
            model_names = list(params.keys())
            actual_model = ""
            for model in model_names:
                if best_model_name==model:
                    actual_model = actual_model+model

            best_params = params[actual_model]
            #Ml-Flow
            mlflow.set_registry_uri("https://dagshub.com/SankritaPatel/StudentPerformanceIndicator.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)