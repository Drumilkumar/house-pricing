"""
train_model.py - Train and save a sample house pricing model.
Run this once to generate model/house_model.pkl before starting the API.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("model", exist_ok=True)

np.random.seed(42)
N = 2000

area       = np.random.randint(500, 5000, N).astype(float)
bedrooms   = np.random.randint(1, 7, N).astype(float)
bathrooms  = np.random.randint(1, 5, N).astype(float)
floors     = np.random.randint(1, 4, N).astype(float)
age        = np.random.randint(0, 50, N).astype(float)
garage     = np.random.randint(0, 2, N).astype(float)
garden     = np.random.randint(0, 2, N).astype(float)
location   = np.random.randint(0, 3, N).astype(float)   # 0=rural,1=suburban,2=urban
distance   = np.random.uniform(0, 50, N)
schools    = np.random.randint(0, 10, N).astype(float)

noise = np.random.normal(0, 15000, N)

price = (
    area * 110
    + bedrooms * 9000
    + bathrooms * 7000
    + floors * 5000
    - age * 600
    + garage * 13000
    + garden * 9000
    + location * 22000
    - distance * 1400
    + schools * 3500
    + noise
).clip(40000, None)

X = np.column_stack([area, bedrooms, bathrooms, floors, age,
                     garage, garden, location, distance, schools])
y = price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"MAE  : ${mean_absolute_error(y_test, preds):,.0f}")
print(f"R²   : {r2_score(y_test, preds):.4f}")

joblib.dump(model, "model/house_model.pkl")
print("Model saved to model/house_model.pkl")
