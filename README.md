# PRODGY_ML_01
Task1
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
Dataset: - https://www.kaggle.com/c/house-prices-advanced-
regression-techniques/data
code
# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sample data: house features (square footage, bedrooms, bathrooms) and prices
# Replace this with your actual dataset
X = np.array([[2000, 3, 2], [1600, 2, 1], [2400, 4, 3], [1416, 3, 2], [3000, 4, 3]])
y = np.array([500000, 350000, 600000, 410000, 700000])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example of predicting a new house price
new_house = np.array([[2500, 4, 3]])
predicted_price = model.predict(new_house)
print(f'Predicted price for new house: ${predicted_price[0]:,.2f}')
