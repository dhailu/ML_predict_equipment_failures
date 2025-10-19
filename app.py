from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

import os
from io import BytesIO
import base64

import matplotlib
matplotlib.use('Agg')  # use non-GUI backend for server
import matplotlib.pyplot as plt

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
                runtime_hours=float(request.form.get('runtime_hours')),
                temperature=float(request.form.get('temperature')),
                vibration_level=float(request.form.get('vibration_level')),
                power_consumption_kw=float(request.form.get('power_consumption_kw')),
                humidity_level=float(request.form.get('humidity_level')),
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


# from flask import Flask, jsonify
# from pymongo import MongoClient
# import pandas as pd
# import joblib

# # -------------------------
# # Flask App
# # -------------------------
# app = Flask(__name__)

# # Load trained ML model
# model = joblib.load("model.pkl")

# # Connect to MongoDB
# uri = "mongodb://user:password@ac-x1.mongodb.net:27017,ac-x2.mongodb.net:27017/maintenance?replicaSet=atlas-xxx-shard-0&ssl=true&authSource=admin&retryWrites=true&w=majority"
# client = MongoClient(uri)

# db = client["maintenance"]
# collection = db["equipment_logs"]

# @app.route("/")
# def home():
#     return {"message": "Equipment Failure Prediction API is running "}

@app.route('/predict_batch', methods=['GET', 'POST'])
def predict_batch():

    try:
        input_df = None
        uploaded_filename = None
        message = None

        if request.method == 'POST':
            action = request.form.get('action')

            # ========== OPTION 1: User uploaded CSV ==========
            if action == 'upload':
                file = request.files.get('file')
                if file and file.filename.lower().endswith('.csv'):
                    try:
                        input_df = pd.read_csv(file)
                        uploaded_filename = file.filename
                        message = f" File '{uploaded_filename}' uploaded successfully."
                    except Exception as e:
                        return jsonify({'error': f"Failed to read uploaded CSV: {e}"}), 400
                else:
                    return jsonify({'error': 'No valid CSV file uploaded.'}), 400

            # ========== OPTION 2: Use default dataset ==========
            elif action == 'default':
                default_file = "artifacts/data.csv"
                if not os.path.exists(default_file):
                    return jsonify({'error': 'Default data.csv not found.'}), 400
                try:
                    input_df = pd.read_csv(default_file)
                    uploaded_filename = "Default dataset (artifacts/data.csv)"
                    message = " Using default dataset."
                except Exception as e:
                    return jsonify({'error': f"Failed to read default CSV: {e}"}), 500

            else:
                return jsonify({'error': 'Invalid action type.'}), 400

            # ===== Normalize column names =====
            input_df.columns = [col.strip().lower() for col in input_df.columns]

            # ===== Run prediction =====
            try:
                predict_pipeline = PredictPipeline()
                predictions = predict_pipeline.predict(input_df)
                input_df["failure_within_7_days"] = predictions
            except Exception as e:
                return jsonify({'error': f"Prediction error: {e}"}), 500

            # ===== Summaries =====
            failed_df = input_df[input_df["failure_within_7_days"] == 1]

            if failed_df.empty:
                return render_template(
                    "predict_batch.html",
                    message=f" No failures predicted within 7 days ({uploaded_filename})."
                )

            failure_counts = failed_df["equipment_type"].value_counts()
            fail_summary = ", ".join([f"{count} {etype}" for etype, count in failure_counts.items()])

            # ===== Build hierarchy =====
            hierarchy = (
                failed_df.groupby(["location", "equipment_type"])["equipment_id"]
                .apply(list)
                .reset_index()
            )
            hierarchy_dict = {}
            for _, row in hierarchy.iterrows():
                loc = row["location"]
                etype = row["equipment_type"]
                eq_list = row["equipment_id"]
                hierarchy_dict.setdefault(loc, {}).setdefault(etype, []).extend(eq_list)

            # ===== Visualization =====
            fig, ax = plt.subplots()
            ax.pie(failure_counts, labels=failure_counts.index, autopct="%1.1f%%")
            ax.set_title("Failures by Equipment Type")
            img = BytesIO()           
            plt.savefig(img, format="png", bbox_inches="tight")
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)
            fig, ax = plt.subplots()
            ax.pie(failure_counts, labels=failure_counts.index, autopct="%1.1f%%")
            ax.set_title("Failures by Equipment Type")

            # Save image to static folder
            img_path = os.path.join('static', 'failure_chart.png')
            plt.savefig(img_path)
            plt.close(fig)

            return render_template(
                "predict_batch.html",
                message=f" {fail_summary} predicted to fail within 7 days ({uploaded_filename}).",
                hierarchy=hierarchy_dict,
                img_data=img_base64
            )

        # Initial load
        return render_template("predict_batch.html")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000) #
    # app.run(debug=True, port=5000)  # For local testing

# import inspect
# print(inspect.getfile(CustomData))
