from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model and label encoders
model = joblib.load('laptop_price_model.joblib')
le_company = joblib.load('label_encoder_company.joblib')
le_typename = joblib.load('label_encoder_typename.joblib')
le_cpu = joblib.load('label_encoder_cpu.joblib')
le_opsys = joblib.load('label_encoder_opsys.joblib')
le_gpu = joblib.load('label_encoder_gpu.joblib')
le_screenresolution = joblib.load('label_encoder_screenresolution.joblib')

# Define input data model
class LaptopFeatures(BaseModel):
    Company: str
    TypeName: str
    Inches: float  # in inches
    ScreenResolution: str
    Cpu: str
    Ram: float  # in GB
    Gpu: str
    OpSys: str
    Weight: float  # in kg

@app.post('/predict_price')
def predict_price(features: LaptopFeatures):
    try:
        # Prepare the input data for prediction
        input_data = np.array([[ 
            le_company.transform([features.Company])[0], 
            le_typename.transform([features.TypeName])[0], 
            features.Inches, 
            le_screenresolution.transform([features.ScreenResolution])[0], 
            le_cpu.transform([features.Cpu])[0], 
            features.Ram, 
            le_gpu.transform([features.Gpu])[0], 
            le_opsys.transform([features.OpSys])[0], 
            features.Weight
        ]])

        # Make prediction
        prediction = model.predict(input_data)
        return {'predicted_price': prediction[0]}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)