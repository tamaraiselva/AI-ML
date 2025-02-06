import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.dot(X, np.array([2, 1])) + 4

# Create a linear regression model
model = LinearRegression().fit(X, y)

# Display model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)