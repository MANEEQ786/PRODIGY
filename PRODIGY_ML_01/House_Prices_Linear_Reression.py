# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('DataSet.csv')

# Check for missing 'Price' values
missing_prices = data[data['Price'].isnull()]

# Handle missing 'Price' values by removing rows with missing prices
data = data.dropna(subset=['Price'])

# Data preprocessing
X = data[["Area", "Room", "Lon", "Lat"]]
y = data["Price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=59)

# Train a linear regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = regression_model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualization
plt.figure(figsize=(12, 6))

# Scatter plot of actual vs. predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, c='b', label='Actual vs. Predicted', alpha=0.5)
plt.title('Actual Prices vs. Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

# Residuals plot
residuals = y_test - y_pred
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
