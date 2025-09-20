import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os
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

# Rolling averages (window sizes 3,5,7,10)
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
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

# ===== Features & Target =====
X = df.drop(["kiln_temp", "timestamp"], axis=1)
y = df["kiln_temp"]

# ===== Time-based Train/Test Split =====
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ===== Convert to DMatrix =====
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ===== XGBoost Parameters =====

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03,
    "max_depth": 5,         # reduce from 6
    "subsample": 0.8,       # slightly reduce
    "colsample_bytree": 0.8,
    "seed": 42,
    "eval_metric": "rmse",
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_child_weight": 5
}

evals = [(dtrain, "train"), (dtest, "eval")]

# ===== Train Model with Early Stopping =====
model = xgb.train(
    params,
    dtrain,
    num_boost_round=3000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# ===== Predict & Evaluate =====
preds = model.predict(dtest)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"RMSE: {rmse:.2f} °C")
print(f"MAE: {mae:.2f} °C")
print(f"R²: {r2:.2f}")

# ===== Save Model =====

output_path = os.path.join(os.getcwd(), f"model.bst")
model.save_model(output_path)
print(f"Model saved at: {output_path}")



# ===== Plot Actual vs Predicted =====
plt.figure(figsize=(15,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(preds, label='Predicted', marker='x')
plt.title("Kiln Temperature Prediction")
plt.xlabel("Time Step")
plt.ylabel("Kiln Temperature")
plt.legend()
plt.show()

# Load saved model
# === Load model again for testing ===
loaded_model = xgb.Booster()
loaded_model.load_model(output_path)

# Build input with correct feature names
sample = pd.DataFrame([{
    "motor_load": 100.0,
    "feeder_rate": 50.0,
    "emissions": 10.0,
    "vibration": 1.0,
    "pressure": 5.0,
    "fuel_rate": 20.0,
    "raw_feed": 200.0,
    "grinding_power": 300.0,
    "hour": 10.0,
    "minute": 22.0,
    "hour_sin": 0.4999999999,
    "hour_cos": -0.8660254037,
    "minute_sin": 0.7431448254,
    "minute_cos": -0.6691306063,
    "prev_temp_1": 601.0,
    "prev_temp_2": 602.0,
    "prev_temp_3": 603.0,
    "prev_temp_4": 604.0,
    "prev_temp_5": 605.0,
    "prev_temp_6": 606.0,
    "rolling_3": 602.0,
    "rolling_5": 602.0,
    "rolling_7": 602.0,
    "rolling_10": 602.0,
    "motor_feeder": 5000.0,
    "motor_fuel": 4.9999975,
    "motor_feeder_fuel": 249.999875,
    "grind_raw_ratio": 1.499999925,
    "motor_emission": 1000.0,
    "fuel_pressure": 100.0,
    "vibration_motor": 100.0
}])

dmatrix = xgb.DMatrix(sample)
pred = loaded_model.predict(dmatrix)
print("Prediction:", pred[0])








FEATURES = [
    "motor_load", "feeder_rate", "emissions", "vibration", "pressure",
    "fuel_rate", "raw_feed", "grinding_power", "hour", "minute",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "prev_temp_1", "prev_temp_2", "prev_temp_3", "prev_temp_4", "prev_temp_5", "prev_temp_6",
    "rolling_3", "rolling_5", "rolling_7", "rolling_10",
    "motor_feeder", "motor_fuel", "motor_feeder_fuel", "grind_raw_ratio",
    "motor_emission", "fuel_pressure", "vibration_motor"
]

# === 2. Load trained model ===
model = xgb.Booster()
model.load_model('model.bst')   # path to your saved model

# === 3. Create test input (example row) ===
samples = [
    [100.0, 50.0, 10.0, 1.0, 5.0, 20.0, 200.0, 300.0, 10.0, 22.0,
     0.5, -0.866, 0.743, -0.669, 601, 602, 603, 604, 605, 606,
     602, 602, 602, 602, 5000, 5.0, 250.0, 1.5, 1000, 100, 100],
    
    [110.0, 55.0, 12.0, 1.2, 6.0, 22.0, 210.0, 320.0, 11.0, 25.0,
     0.42, -0.91, 0.84, -0.54, 590, 591, 592, 593, 594, 595,
     591, 591, 591, 591, 6050, 5.0, 275.0, 1.52, 1320, 132, 132]
]

df = pd.DataFrame(samples, columns=FEATURES)

dmatrix = xgb.DMatrix(df)
preds = model.predict(dmatrix)

for i, p in enumerate(preds):
    print(f"Row {i+1} → Prediction: {p:.2f}")

# ===== Plot Actual vs Predicted =====
plt.figure(figsize=(15,5))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(preds, label='Predicted', marker='x')
plt.title("Kiln Temperature Prediction")
plt.xlabel("Time Step")
plt.ylabel("Kiln Temperature")
plt.legend()
plt.show()



