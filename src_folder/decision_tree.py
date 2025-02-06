import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
data = pd.read_csv('./train.csv')  # Adjust the path if necessary
print(data.head())

# Handle missing values
# Fill missing values for numeric columns with their mean
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns  # Get numeric columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Replace NaN with mean

# Fill missing values for categorical columns with their mode
categorical_cols = data.select_dtypes(include=['object']).columns  # Get categorical columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)  # Fill with mode (most common value)

# Convert categorical variables to dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data.drop(['SalePrice'], axis=1)  # Features
y = data['SalePrice']  # Target variable


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)


# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.2f}')


plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, max_depth=3)
plt.title('Decision Tree Visualization')
plt.show()


# Getting feature importances
importances = model.feature_importances_

# Creating a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualizing feature importances
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()