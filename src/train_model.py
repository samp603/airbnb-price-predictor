"""
train_model.py - Baseline model training for Airbnb Price Predictor

This script:
- Loads the cleaned dataset
- Splits data into train/test sets
- Trains a Linear Regression model
- Outputs RMSE and R² as baseline metrics

TODOs:
- Try Random Forest or XGBoost models and compare performance
- Save model to disk for reuse
- Add feature importance visualization
- Add cross-validation (e.g., k-fold)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_baseline_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(['price'], axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Baseline Linear Regression Model:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")


if __name__ == '__main__':
    train_baseline_model('../data/cleaned_data.csv')
