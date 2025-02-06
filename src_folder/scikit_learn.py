import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# Create a linear regression model
model = LinearRegression().fit(X, y)

# Predict using the model
predictions = model.predict(np.array([[3, 5]]))
print("Predictions:", predictions)