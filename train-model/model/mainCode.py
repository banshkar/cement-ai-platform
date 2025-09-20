import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ===== Load Data =====
df = pd.read_csv("/Users/jitendra_banshkar/Desktop/2025/data/cement-ai-platform/ingest-service/cement_synthetic_good_quality.csv")

# ===== Feature Engineering =====
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

# Cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)

# Lag features (previous 1–6 steps)
for lag in range(1, 7):
    df[f'prev_temp_{lag}'] = df['kiln_temp'].shift(lag)

# Rolling averages (window sizes 3, 5, 7, 10)
for window in [3, 5, 7, 10]:
    df[f'rolling_{window}'] = df['kiln_temp'].rolling(window).mean().shift(1)

# Interaction features
df['motor_feeder'] = df['motor_load'] * df['feeder_rate']
df['motor_fuel'] = df['motor_load'] / (df['fuel_rate'] + 1e-5)
df['motor_feeder_fuel'] = df['motor_load'] * df['feeder_rate'] / (df['fuel_rate'] + 1e-5)
df['grind_raw_ratio'] = df['grinding_power'] / (df['raw_feed'] + 1e-5)
df['motor_emission'] = df['motor_load'] * df['emissions']
df['fuel_pressure'] = df['fuel_rate'] * df['pressure']
df['vibration_motor'] = df['vibration'] * df['motor_load']

# Fill NaNs
df = df.bfill().ffill()

# ===== Features & Target =====
X = df.drop(["kiln_temp", "timestamp"], axis=1)
y = df["kiln_temp"]

# Save feature columns for reference
feature_cols = X.columns.tolist()
with open("feature_cols.txt", "w") as f:
    f.write("\n".join(feature_cols))
print(f"✅ Saved {len(feature_cols)} training features to feature_cols.txt")

# ===== Time-based Train/Test Split =====
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===== Convert to DMatrix (with feature names) =====
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

print("DMatrix shapes:", dtrain.num_row(), dtrain.num_col(), dtest.num_row(), dtest.num_col())
print("Feature names in DMatrix:", dtrain.feature_names)

# ===== XGBoost Parameters =====
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "eval_metric": "rmse",
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_child_weight": 5
}

# ===== Train Model =====
model = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    evals=[(dtrain, "train"), (dtest, "eval")],
    early_stopping_rounds=50,
    verbose_eval=50
)

# ===== Predict & Evaluate =====
preds = model.predict(dtest)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"RMSE: {rmse:.2f} °C")
print(f"MAE: {mae:.2f} °C")
print(f"R²: {r2:.2f}")

# ===== Save Model for Vertex AI =====
model.save_model("model.bst")       # Binary format
model.save_model("model.json")      # JSON format (Vertex AI can load this)

print("✅ Model saved in both .bst and .json formats with feature names included")

# ===== Reload & Predict (Sanity Check) =====
model_reload = xgb.Booster()
model_reload.load_model("model.bst")
dtest_reload = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
booster = xgb.Booster()
booster.load_model("model.bst")
print("Feature names in booster:", booster.feature_names)

preds_reload = model_reload.predict(dtest_reload)
print("✅ Sanity check predictions:", preds_reload[:10])
