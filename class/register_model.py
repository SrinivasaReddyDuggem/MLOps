import mlflow
from mlflow import MlflowClient # Imports a client to manage registered models, versions, and aliases.

# 1. Connect to the existing MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# 2. Locate the experiment from Topic 3
experiment = mlflow.get_experiment_by_name( # Searches MLflow for an experiment using its name.
    "Retail_Demand_Forecasting"
)

if experiment is None:
    raise RuntimeError("Run experiment.py first.")

# 3. Find the run with the lowest MAE
runs = mlflow.search_runs( # Searches previously recorded experiment runs.
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.mae >= 0",
    order_by=["metrics.mae ASC"],
    max_results=1
)

if runs.empty:
    raise RuntimeError("No evaluated runs found.")

best_run_id = runs.iloc[0]["run_id"] # Retrieves the unique ID of the best experiment run.
best_mae = runs.iloc[0]["metrics.mae"] # Retrieves the best run's average prediction error.

# 4. Find the logged model belonging to that run
logged_models = mlflow.search_logged_models( # Searches for models previously saved in MLflow.
    experiment_ids=[experiment.experiment_id],
    output_format="list"
)

matching_models = [
    model for model in logged_models
    if model.source_run_id == best_run_id
]

if not matching_models:
    raise RuntimeError("No model found for the best run.")

best_model = matching_models[0] # Selects the first saved model matching the best run.

# 5. Register the previously saved model i.e. Register the best model
registered = mlflow.register_model(
    model_uri=f"models:/{best_model.model_id}",
    name="Retail_Demand_Model"
)

# 6. Assign a champion alias
client.set_registered_model_alias(
    name="Retail_Demand_Model",
    alias="champion",
    version=registered.version
)

print("Best run ID:", best_run_id)
print("Best MAE:", round(best_mae, 3))
print("Registered version:", registered.version)
print("Champion alias assigned")
