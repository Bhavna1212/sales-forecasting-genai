from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Model file path
model_path = "sales_forecast_model.pkl"

# Check if model file exists and load it
if not os.path.exists(model_path):
    logging.error(f"Model file {model_path} not found!")
    raise FileNotFoundError(f"Model file {model_path} not found!")

try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise e

@app.route("/", methods=["GET"])
def home():
    logging.info("Home route accessed.")
    return jsonify({"message": "Generative AI Sales Forecast API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logging.info("Received prediction request.")

        # Get JSON request data
        data = request.json
        logging.info(f"Request Data: {data}")

        if not data or "date" not in data:
            logging.warning("Missing 'date' field in request.")
            return jsonify({"error": "Missing 'date' field"}), 400

        # Convert date input
        date_input = data.get("date")
        future_df = pd.DataFrame({"ds": [pd.to_datetime(date_input)]})
        logging.info(f"Input DataFrame: {future_df}")

        # Predict
        forecast = model.predict(future_df)
        prediction = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

        logging.info(f"Prediction: {prediction}")
        return jsonify({"forecast": prediction})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
