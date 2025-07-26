"""
train_model.py - Model training for Airbnb Price Predictor

This script:
- Loads the cleaned dataset
- Splits data into train/test sets
- Trains Linear Regression, Random Forest, and XGBoost models
- Outputs RMSE and R² metrics for comparison
- Saves all models to disk

TODOs:
- Add feature importance visualization
- Add cross-validation (e.g., k-fold)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def evaluate_model(name, model, X_test, y_test, results):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "R²": r2}
    print(f"{name}:\n  RMSE: {rmse:.2f}\n  R²: {r2:.2f}\n")


def train_models(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(['price'], axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate_model("Linear Regression", lr, X_test, y_test, results)
    models["linear_regression_model.pkl"] = lr

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test, results)
    models["random_forest_model.pkl"] = rf

    # XGBoost
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)
    evaluate_model("XGBoost", xgb, X_test, y_test, results)
    models["xgboost_model.pkl"] = xgb

    # Save models
    os.makedirs("../models", exist_ok=True)
    for filename, model in models.items():
        joblib.dump(model, f"../models/{filename}")

    # Summary
    print("All models trained and saved successfully.")
    print(pd.DataFrame(results).T)


if __name__ == '__main__':
    train_models('../data/engineered_features.csv')
