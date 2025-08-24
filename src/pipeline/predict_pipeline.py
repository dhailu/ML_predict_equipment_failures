import sys
import pandas as pd
from src.exception import CustomeException
from src.utils import load_object
import joblib

class PredictPipeline:
    def __init__(self):
        self.model = joblib.load("artifacts/model.pkl")
        # pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl' 

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomeException(e, sys)  

# {
#   "error": "'CustomData' object has no attribute 'get_data_as_dataframe'"
# }   
# {
#   "error": "Error occured in python script name [c:\\Users\\Dejene\\Documents\\Git_comany\\ML_predict_equipment_failures\\src\\pipeline\\predict_pipeline.py] line number [21] error message [X has 822 features, but AdaBoostRegressor is expecting 2619 features as input.]"
# }
class CustomData:
    def __init__(self, 
                equipment_id: str,
                equipment_type: str,
                location: str,
                # install_date,
                # last_service_date: str,
                # next_scheduled_service: str,
                service_priority: str,
                age_days: int,
                runtime_hours: float,
                temperature: float,
                vibration_level: float,
                power_consumption_kw: float,
                humidity_level: float,
                error_codes_count: int,
                manual_override: int,
                downtime_last_30d: int 
                ):
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type        
        self.location = location
        # self.install_date = install_date    
        # self.last_service_date = last_service_date
        # self.next_scheduled_service = next_scheduled_service
        self.service_priority = service_priority
        self.age_days = age_days
        self.runtime_hours = runtime_hours
        self.temperature = temperature
        self.vibration_level = vibration_level
        self.power_consumption_kw = power_consumption_kw
        self.humidity_level = humidity_level
        self.error_codes_count = error_codes_count
        self.manual_override = manual_override
        self.downtime_last_30d = downtime_last_30d

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "equipment_id": [self.equipment_id],
                "equipment_type": [self.equipment_type],
                "location": [self.location],
                # "install_date": [self.install_date],
                # "last_service_date": [self.last_service_date],
                # "next_scheduled_service": [self.next_scheduled_service],
                "service_priority": [self.service_priority],
                "age_days": [self.age_days],
                "runtime_hours": [self.runtime_hours],
                "temperature": [self.temperature],
                "vibration_level": [self.vibration_level],
                "power_consumption_kw": [self.power_consumption_kw],
                "humidity_level": [self.humidity_level],
                "error_codes_count": [self.error_codes_count],
                "manual_override": [self.manual_override],
                "downtime_last_30d": [self.downtime_last_30d]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomeException(e, sys)

    
