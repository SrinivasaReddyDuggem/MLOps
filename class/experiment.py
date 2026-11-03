# 1. Generate 240 days of simulated retail sales data.
# 2. Train three Random Forest forecasting models.
# 3. Measure how accurately each model predicts sales.
# 4. Track model settings and errors using MLflow.
# 5. Save all three trained models.
# 6. Identify the model with the lowest MAE.

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from mlflow.models import infer_signature # Imports a function to identify model input and output formats.

# STEP 1: Connect to our local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# STEP 2: Create or select the experiment
mlflow.set_experiment("Retail_Demand_Forecasting") # Creates or selects an experiment to store all model results.

# STEP 3: Reuse the same dataset from Topic 2
rng = np.random.default_rng(42)

days = np.arange(1, 241)
day_of_week = (days - 1) % 7
promotion = ((days % 17) < 3).astype(int)

sales = (
    95
    + 15 * (day_of_week >= 5)
    + 30 * promotion
    + rng.normal(0, 5, len(days))
)

sales = np.maximum(0, np.rint(sales)).astype(int)

df = pd.DataFrame({
    "day": days,
    "day_of_week": day_of_week,
    "promotion": promotion,
    "sales": sales
})

df["sales_lag_7"] = df["sales"].shift(7)
df = df.dropna().copy()

features = [
    "day_of_week",
    "promotion",
    "sales_lag_7"
]

train = df[df["day"] <= 200]
test = df[df["day"] > 200]

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

# STEP 4: Define three model configurations
configurations = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 8}
]

# STEP 5: Train and track each configuration
results = []

for config in configurations:

    run_name = (
        f"RF_Trees_{config['n_estimators']}"
        f"_Depth_{config['max_depth']}"
    )

    with mlflow.start_run(run_name=run_name):

        # Create and train the model
        model = RandomForestRegressor(
            **config,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Generate predictions and evaluate
        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test, predictions
        )

        # Record the experiment settings
        mlflow.log_params(config) # Saves the model settings, including number of trees and depth.

        # Record the evaluation result
        mlflow.log_metric("mae", mae) # Saves the model's prediction error in MLflow.

        # Add useful descriptive metadata
        mlflow.set_tag(
            "use_case", "Retail Demand Forecasting" # Adds a descriptive label explaining the business use case.
        )

        # Define the model's input and output schema
        signature = infer_signature( # Records the expected input columns and prediction output format.
            X_test, predictions
        )

        # Save the model in MLflow
        model_info = mlflow.sklearn.log_model( # Saves the trained Scikit-learn model as an MLflow artifact.
            sk_model=model,
            name="demand_model",
            signature=signature,
            input_example=X_train.head(3) # Saves three example input rows to demonstrate how the model is used.
        )

        # Remember the results for comparison i.e. Store results for comparison
        results.append({
            "run_name": run_name,
            "mae": mae,
            "model_uri": model_info.model_uri # Stores the MLflow location where the model was saved.
        })

        print(f"{run_name} | MAE: {mae:.3f}")

# STEP 6: Identify the lowest-error model
best = min(results, key=lambda x: x["mae"])

print("\nBEST MODEL")
print("Run:", best["run_name"])
print("MAE:", round(best["mae"], 3))
print("Model URI:", best["model_uri"])
