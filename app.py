from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Check if model file exists
model_path = "sales_forecast_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

# Load trained model
model = joblib.load(model_path)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Generative AI Sales Forecast API is Running!"})

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

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
