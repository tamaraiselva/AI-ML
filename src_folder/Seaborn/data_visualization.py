import seaborn as sns
import matplotlib.pyplot as plt


## Plotting with Seaborn

# Load sample dataset
data = sns.load_dataset('iris')

sns.scatterplot(x='sepal_length', y='sepal_width', data=data)
plt.title('Sepal Length vs Sepal Width')
plt.show()



## Customizing Seaborn Plots

sns.set(style='whitegrid')
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=data)
plt.title('Scatter Plot by Species')
plt.show()


## Seaborn Distribution Plots

sns.histplot(data['sepal_length'], kde=True, color='purple')
plt.title('Sepal Length Distribution with KDE')
plt.show()


