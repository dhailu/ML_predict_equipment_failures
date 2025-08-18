import pandas as pd
import sys
import os
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")   

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostClassifier": CatBoostClassifier(verbose=0)
            }
            params = {
            "LinearRegression": {},
            "DecisionTreeRegressor": {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            },
            "RandomForestRegressor": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            "GradientBoostingRegressor": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]  
            },
            "AdaBoostRegressor": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 1.0]   
            
            },
        
            "KNeighborsRegressor": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            "XGBRegressor": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            },
            "CatBoostClassifier": {
                'iterations': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7],
                'l2_leaf_reg': [1, 3, 5]
            }
            }   
            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                model_report[model_name] = {
                    "r2_score": r2,'\n'
                    "mean_absolute_error": mae,'\n'
                    "mean_squared_error": mse
                }
            
            best_model_name = max(model_report, key=lambda x: model_report[x]["r2_score"])
            best_model = models[best_model_name]
            
            logging.info(f"Best model found: {best_model_name} with R2 score: {model_report[best_model_name]['r2_score']}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            return best_model_name, model_report[best_model_name]

        except Exception as e:
            raise CustomeException(e, sys)  