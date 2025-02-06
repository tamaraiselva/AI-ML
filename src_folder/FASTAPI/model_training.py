import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
data = pd.read_csv('laptop_data.csv')

# Print the columns to see what was read
print("Columns in the dataset:", data.columns.tolist())

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check if the required columns exist
required_columns = ['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Gpu', 'OpSys', 'Weight', 'Price']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"The required column '{col}' is not present in the CSV file.")

# Encode categorical variables
le_company = LabelEncoder()
le_typename = LabelEncoder()
le_cpu = LabelEncoder()
le_opsys = LabelEncoder()
le_gpu = LabelEncoder()
le_screenresolution = LabelEncoder()

data['Company'] = le_company.fit_transform(data['Company'])
data['TypeName'] = le_typename.fit_transform(data['TypeName'])
data['Cpu'] = le_cpu.fit_transform(data['Cpu'])
data['OpSys'] = le_opsys.fit_transform(data['OpSys'])
data['Gpu'] = le_gpu.fit_transform(data['Gpu'])
data['ScreenResolution'] = le_screenresolution.fit_transform(data['ScreenResolution'])

# Convert 'Ram' and 'Weight' to numeric values
data['Ram'] = data['Ram'].str.replace('GB', '').astype(float)  # Assuming ram is in "8GB" format
data['Weight'] = data['Weight'].str.replace('kg', '').str.strip().astype(float)  # Ensure weight is numeric

# Prepare features and target variable
X = data[['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Gpu', 'OpSys', 'Weight']]
y = data['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(model, 'laptop_price_model.joblib')
joblib.dump(le_company, 'label_encoder_company.joblib')
joblib.dump(le_typename, 'label_encoder_typename.joblib')
joblib.dump(le_cpu, 'label_encoder_cpu.joblib')
joblib.dump(le_opsys, 'label_encoder_opsys.joblib')
joblib.dump(le_gpu, 'label_encoder_gpu.joblib')
joblib.dump(le_screenresolution, 'label_encoder_screenresolution.joblib')
