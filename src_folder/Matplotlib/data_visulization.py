import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]

plt.plot(x, y)
plt.title('Simple Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


plt.plot(x, y, color='green', linestyle='--', marker='o')
plt.title('Customized Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()


data = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
plt.hist(data, bins=5, color='blue', edgecolor='black')
plt.title('Data Distribution')
plt.show()


## Visualizing Model Performance Metrics

epochs = range(1, 11)
accuracy = [0.65, 0.72, 0.75, 0.78, 0.80, 0.82, 0.83, 0.85, 0.86, 0.87]

plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


## Plotting Confusion Matrix

# Example confusion matrix data
cm = confusion_matrix([1, 1, 0, 0], [1, 0, 1, 0])

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



## Plotting ROC Curve

# Example model predictions
fpr, tpr, thresholds = roc_curve([1, 1, 0, 0], [0.9, 0.8, 0.4, 0.2])

plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()