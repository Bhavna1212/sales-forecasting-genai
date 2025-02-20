from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, force CPU
import tensorflow as tf
from transformers import pipeline
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained model (assuming Prophet for forecasting)
model = joblib.load("sales_forecast_model.pkl")  # Ensure this file is in your repo

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Generative AI Sales Forecast API!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input with 'date' field
        date_input = data.get("date")

        if not date_input:
            return jsonify({"error": "Missing date field"}), 400

        future_df = pd.DataFrame({"ds": [pd.to_datetime(date_input)]})
        forecast = model.predict(future_df)
        prediction = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

        return jsonify({"forecast": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
