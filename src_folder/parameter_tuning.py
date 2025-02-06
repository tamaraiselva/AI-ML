## Grid Search

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting the dataset 
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
print("Best parameters from Grid Search:", grid_search.best_params_)



## Random Search

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split,cross_val_score 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
data = load_iris() 
X = data.data 
y = data.target 
#splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using Random Search
random_param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None] + list(np.arange(10, 30, 5)),
    'min_samples_split': np.arange(2, 20, 2)
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=random_param_dist,n_iter=100, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Best parameters from Random Search
print("Best parameters from Random Search:", random_search.best_params_)



## Model Evaluation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Evaluate the model with best parameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Accuracy with Grid Search tuning:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



## Visualizing Results

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Grid Search)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


## Visualizing the Confusion Matrix for the best model from Random Search

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Random Search)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()