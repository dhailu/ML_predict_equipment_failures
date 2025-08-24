import pandas as pd
import sys  
import os 
import numpy
from src.exception import CustomeException

# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from sklearn.model_selection import GridSearchCV
import pickle
      

def save_object(obj, file_path):
    """
    Save an object to a file using pandas.
    
    Parameters:
    obj (object): The object to save.
    file_path (str): The path where the object will be saved.
    """
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        pd.to_pickle(obj, file_path)
        print(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise CustomeException(f"Error saving object: {e}", sys) 

# def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para = param[list(models.keys())[i]] # if param else {}
#             gs= GridSearchCV(
#                 model,
#                 para,
#                 cv=3,
#                 # n_jobs=-1,
#                 # verbose=2
#             ) #if para else model
#             gs.fit(X_train, y_train)
#             # model = gs.best_estimator_ if para else model

#             model.set_params(**gs.best_params_) # Train model
#             model.fit(X_train, y_train)  
 
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test) 
#             train_model_score = r2_score(y_train, y_train_pred)
#             test_model_score = r2_score(y_test, y_test_pred)    
#             report[list(models.keys())[i]] = test_model_score



#         return report
# def evaluate_models(y_true, y_pred):
def evaluate_models(X_train, y_train, X_test, y_test, models, param, threshold=0.5):
    report = {}
    try:
        for name, model in models.items():
            para = param.get(name, {})  # get parameters if available

            # GridSearchCV
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            # Best model
            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            # Default predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Default metrics (threshold = 0.5)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]) \
                    if hasattr(best_model, "predict_proba") else None

            # ---- Threshold adjustment ----
            custom_precision, custom_recall, custom_f1 = None, None, None
            if hasattr(best_model, "predict_proba"):
                y_probs = best_model.predict_proba(X_test)[:, 1]
                y_test_custom = (y_probs >= threshold).astype(int)

                custom_precision = precision_score(y_test, y_test_custom)
                custom_recall = recall_score(y_test, y_test_custom)
                custom_f1 = f1_score(y_test, y_test_custom)

            # Collect results
            report[name] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "precision_default": precision,
                "recall_default": recall,
                "f1_default": f1,
                "roc_auc": roc_auc,
                f"precision_thr_{threshold}": custom_precision,
                f"recall_thr_{threshold}": custom_recall,
                f"f1_thr_{threshold}": custom_f1
            }

        return report

    except Exception as e:
        raise CustomeException(e, sys)
###

 
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomeException(e, sys)