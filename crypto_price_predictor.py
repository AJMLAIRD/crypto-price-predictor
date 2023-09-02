import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load cryptocurrency price data (replace with your dataset)
# Sample data should contain 'Date' and 'Price' columns
data = pd.read_csv('crypto_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Feature engineering: create a feature for the number of days since the start
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Define the features and target variable
X = data[['Days']].values
y = data['Price'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Days Since Start')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
