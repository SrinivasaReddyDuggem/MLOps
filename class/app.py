import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Connect to the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Load the registered champion model once
model = mlflow.sklearn.load_model(
    "models:/Retail_Demand_Model@champion"
)

# 3. Create the API application
app = FastAPI(title="Retail Demand Prediction API")

# 4. Define the expected input
class DemandInput(BaseModel):
    day_of_week: int = Field(ge=0, le=6)
    promotion: int = Field(ge=0, le=1)
    sales_lag_7: float = Field(ge=0)

# 5. Create a health check endpoint
@app.get("/health") # Creates a GET endpoint at /health to check the application.
def health():
    return {"status": "ok"}

# Example: Opening http://127.0.0.1:8000/health returns:
# {"status": "ok"}

# 6. Create a prediction endpoint
@app.post("/predict") # Creates a POST endpoint that accepts prediction requests.
def predict(data: DemandInput):

    # Convert JSON input into model features
    features = pd.DataFrame([{
        "day_of_week": data.day_of_week,
        "promotion": data.promotion,
        "sales_lag_7": data.sales_lag_7
    }])

    # Generate prediction
    prediction = model.predict(features)[0]

    # Return JSON response
    return {
        "predicted_demand": round(
            max(0, float(prediction)), 2
        ),
        "unit": "units"
    }
