from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['target'] = housing.target

# Display the first few rows of the dataset
print(data.head())

# Split the data into training and testing sets 
X = data.drop('target', axis=1) 
y = data['target'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Create and train the model 
model = LinearRegression() 
model.fit(X_train, y_train) 
# Make predictions 
y_pred = model.predict(X_test) 
# Evaluate the model 
mse = mean_squared_error(y_test, y_pred) 
r2 = r2_score(y_test, y_pred) 
print(f"Mean Squared Error: {mse}") 
print(f"R^2 Score: {r2}") 

# Plot true vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Diagonal line
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.grid()
plt.show()


# Plot distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(data['target'], bins=30, kde=True)
plt.xlabel('House Prices')
plt.title('Distribution of House Prices')
plt.grid()
plt.show()


# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.grid()
plt.show()


# Plot feature importance
plt.figure(figsize=(12, 6))
features = X.columns
importance = model.coef_
sns.barplot(x=importance, y=features)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.grid()
plt.show()