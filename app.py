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
        # Expecting JSON input with 'date' field
        data = request.get_json()
        if not data or "date" not in data:
            return jsonify({"error": "Missing 'date' field in JSON request"}), 400

        try:
            date_input = pd.to_datetime(data["date"])  # Convert to datetime
        except Exception:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        # Create DataFrame for prediction
        future_df = pd.DataFrame({"ds": [date_input]})
        forecast = model.predict(future_df)

        # Extract forecast values
        prediction = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")

        return jsonify({"forecast": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Render automatically sets PORT, so no need to run Flask manually
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
