import pandas as pd
import sys  
import os 
import numpy
from src.exception import CustomeException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
 
      

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

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            # model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomeException(e, sys)
###





    

    
# def load_object(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return pickle.load(file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)