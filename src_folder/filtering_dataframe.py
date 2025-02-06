import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Boston']}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Filter DataFrame for Age greater than 28
filtered_df = df[df['Age'] > 28]
print("Filtered DataFrame:\n", filtered_df)