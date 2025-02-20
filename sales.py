import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random

# Generate realistic sales data
def generate_sales_data(n=1000):
    np.random.seed(42)
    random.seed(42)
    
    products = ['Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Printer']
    business_units = ['Retail', 'Wholesale', 'E-commerce']
    geos = ['North America', 'Europe', 'Asia']
    countries = {'North America': ['USA', 'Canada'],
                 'Europe': ['Germany', 'UK', 'France'],
                 'Asia': ['India', 'China', 'Japan']}
    
    territories = {'USA': ['East', 'West', 'North', 'South'],
                   'Canada': ['Ontario', 'Quebec'],
                   'Germany': ['Berlin', 'Bavaria'],
                   'UK': ['London', 'Manchester'],
                   'France': ['Paris', 'Lyon'],
                   'India': ['Delhi', 'Mumbai'],
                   'China': ['Beijing', 'Shanghai'],
                   'Japan': ['Tokyo', 'Osaka']}
    
    data = []
    for i in range(n):
        product = random.choice(products)
        business_unit = random.choice(business_units)
        geo = random.choice(geos)
        country = random.choice(countries[geo])
        territory = random.choice(territories[country])
        date = pd.to_datetime('2020-01-01') + pd.DateOffset(days=random.randint(0, 1460))
        sales = max(50, np.random.normal(500, 150))  # Ensure sales are always positive
        
        data.append([date, product, business_unit, geo, country, territory, sales])
    
    df = pd.DataFrame(data, columns=['Date', 'Product', 'Business Unit', 'Geography', 'Country', 'Territory', 'Sales'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

# Generate and display sample data
df = generate_sales_data()
print(df.head())

# Normalize Sales Data
scaler = MinMaxScaler(feature_range=(0, 1))
df['Sales_Scaled'] = scaler.fit_transform(df[['Sales']])

# Convert time series data into sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Set sequence length
seq_length = 10
sales_data = df['Sales_Scaled'].values
X, y = create_sequences(sales_data, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Adding input feature dimension
y_tensor = torch.FloatTensor(y).unsqueeze(-1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1])  # Use the last time step's output
        return predictions

# Initialize model, loss, optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
def train_model(model, X_train, y_train, epochs=50):
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

train_model(model, X_tensor, y_tensor)

# Predict future sales
model.eval()
y_pred = model(X_tensor).detach().numpy()

# Convert predictions back to original scale
y_pred_actual = scaler.inverse_transform(y_pred)

# Plot results
plt.figure(figsize=(10,5))
plt.plot(df['Date'][seq_length:], df['Sales'][seq_length:], label='Actual Sales', color='blue')
plt.plot(df['Date'][seq_length:], y_pred_actual, label='Predicted Sales', linestyle='dashed', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask is running!"

# Example API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Expect JSON input
    return jsonify({"prediction": "Your model output here"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
