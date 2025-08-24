import pandas as pd
import sys
import os
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    # GradientBoostingRegressor
     AdaBoostRegressor
)
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object, evaluate_models

import matplotlib.pyplot as plt
import seaborn as sns

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
                "LogisticRegression": LogisticRegression(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "SVC": SVC(probability=True),  # probability=True for ROC-AUC
                "KNeighbors": KNeighborsClassifier(),
                "NaiveBayes": GaussianNB(),
                "DecisionTree": DecisionTreeClassifier()
            }
            ## Define hyperparameters for each model
            params = {
                    "LogisticRegression": {
                        "penalty": ["l1", "l2", "elasticnet", None],
                        "C": [0.01, 0.1, 1, 10],
                        "solver": ["saga", "lbfgs"]
                    },
                    "RandomForest": {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 5, 10],
                        "min_samples_split": [2, 5]
                    },
                    "GradientBoosting": {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5]
                    },
                    "AdaBoost": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 1]
                    },
                    "SVC": {
                        "C": [0.1, 1, 10],
                        "kernel": ["linear", "rbf"]
                    },
                    "KNeighbors": {
                        "n_neighbors": [3, 5, 7],
                        "weights": ["uniform", "distance"]
                    },
                    "NaiveBayes": {},  # no params
                    "DecisionTree": {
                        "max_depth": [None, 5, 10],
                        "min_samples_split": [2, 5, 10]
                    }
                }
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                threshold=0.3  #  try different thresholds here
            )

            # print(model_report)
            
            # best_model_name = max(model_report, key=lambda x: model_report[x]["r2_score"])
            # best_model = models[best_model_name]
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_accuracy'])
            best_model = models[best_model_name]

            logging.info(f'Best model found: {best_model_name} with accuracy score: {model_report[best_model_name]["test_accuracy"]}')
            logging.info(f'best model is {best_model}')
            
            # logging.info(f"Best model found: {best_model_name} with R2 score: {model_report[best_model_name]['r2_score']}")
            # logging.info(f'best model is {best_model}')
            # best_model = AdaBoostClassifier()
            best_model.fit(X_train, y_train)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                        
            return best_model_name, model_report[best_model_name]

        except Exception as e:
            raise CustomeException(e, sys)  
        

# def plot(precisions, recalls, thresholds):
#     plt.figure(figsize=(8, 6))
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
#     plt.xlabel("Threshold")
#     plt.title("Precision and Recall vs Decision Threshold")
#     plt.legend(loc="best")
#     plt.grid()
#     plt.show()
#     plt.savefig('artifacts/precision_recall_vs_threshold.png')
#     plt.close()
# # Example: choose threshold = 0.3 (instead of 0.5)
# threshold = 0.3
# y_pred_custom = (y_probs >= threshold).astype(int)

# print("Classification Report with threshold=0.3")
# print(classification_report(y_test, y_pred_custom))

