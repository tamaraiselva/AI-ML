import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('./train.csv')


# Display basic information about the dataset
print(data.info())
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0])

# Fill missing values (example: mean imputation for LotFrontage)
data['LotFrontage'].fillna(data['LotFrontage'].mean(), inplace=True)

# For categorical features, fill with mode
data['MasVnrType'].fillna(data['MasVnrType'].mode()[0], inplace=True)


# Create a bar plot for missing values before handling
plt.figure(figsize=(10, 6))
missing_values[missing_values > 0].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Missing Values by Feature')
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.show()

# Recheck missing values after imputation
missing_values_after = data.isnull().sum()
plt.figure(figsize=(10, 6))
missing_values_after[missing_values_after > 0].plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Missing Values After Imputation')
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.show()


# Normalize numerical features
scaler = MinMaxScaler()
data[['LotFrontage']] = scaler.fit_transform(data[['LotFrontage']])


# Plotting before and after normalization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data['LotFrontage'], bins=30, color='skyblue', edgecolor='black')
plt.title('LotFrontage Distribution (Before Normalization)')
plt.xlabel('LotFrontage')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(data['LotFrontage'], bins=30, color='lightgreen', edgecolor='black')
plt.title('LotFrontage Distribution (After Normalization)')
plt.xlabel('LotFrontage')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Encoding categorical variables
data = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)

# Check the column names
print("Columns in the dataset:")
print(data.columns)

# Strip whitespace from headers (if needed)
data.columns = data.columns.str.strip()

# Check if 'Neighborhood' column exists
if 'Neighborhood' in data.columns:
    # Visualizing the distribution of the 'Neighborhood' feature
    plt.figure(figsize=(10, 6))
    data['Neighborhood'].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
    plt.title('Distribution of Neighborhoods')
    plt.xlabel('Neighborhood')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("The 'Neighborhood' column does not exist in the dataset. Available columns are:")
    print(data.columns)


    