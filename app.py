from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import joblib

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template ('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
        
    else:
        try:
            data = CustomData(
                equipment_id=request.form.get('equipment_id'),
                equipment_type=request.form.get('equipment_type'),
                location=request.form.get('location'),
                # install_date=request.form.get('install_date'),
                # last_service_date=request.form.get('last_service_date'),
                # next_scheduled_service=request.form.get('next_scheduled_service'),
                service_priority=request.form.get('service_priority'),
                age_days=int(request.form.get('age_days')),
                runtime_hours=int(request.form.get('runtime_hours')),
                temperature=int(request.form.get('temperature')),
                vibration_level=int(request.form.get('vibration_level')),
                power_consumption_kw=int(request.form.get('power_consumption_kw')),
                humidity_level=int(request.form.get('humidity_level')),
                error_codes_count=int(request.form.get('error_codes_count')),
                manual_override=int(request.form.get('manual_override')),
                downtime_last_30d=float(request.form.get('downtime_last_30d'))
            )

            
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before prediction")

            predict_pipeline = PredictPipeline()
            print("Middle of prediction")
            prediction = predict_pipeline.predict(pred_df)
            print("After prediction")

            # return render_template('home.html', prediction_text=f'Predicted Equipment Failure: {prediction[0]}')
            return render_template('home.html', prediction = prediction[0])   

        except Exception as e:
            return jsonify({'error': str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000) #
    # app.run(debug=True, port=5000)  # For local testing

# import inspect
# print(inspect.getfile(CustomData))
