
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# STEP 1: Generate 240 days of retail sales
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

# STEP 2: Create a dataset
df = pd.DataFrame({
    "day": days,
    "day_of_week": day_of_week,
    "promotion": promotion,
    "sales": sales
})

# STEP 3: Create a historical sales feature
df["sales_lag_7"] = df["sales"].shift(7)
df = df.dropna().copy()

# STEP 4: Define model inputs and target
features = [
    "day_of_week",
    "promotion",
    "sales_lag_7"
]

target = "sales"

# STEP 5: Split chronologically
train = df[df["day"] <= 200]
test = df[df["day"] > 200]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# STEP 6: Create the ML model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# STEP 7: Train
model.fit(X_train, y_train)

# STEP 8: Predict on unseen dates
predictions = model.predict(X_test)

# STEP 9: Evaluate model performance
mae = mean_absolute_error(
    y_test,
    predictions
)

print(f"Mean Absolute Error: {mae:.2f} units")

# STEP 10: Display predictions
results = pd.DataFrame({
    "actual_sales": y_test.to_numpy(),
    "predicted_sales": predictions.round(1)
})

print(results.head())

# STEP 11: Save reusable project artifacts
output = Path("artifacts")
output.mkdir(exist_ok=True)

joblib.dump(model, output / "demand_model.joblib")
joblib.dump(features, output / "features.joblib")

print("Model saved successfully!")
