import os
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def simplify_categories(df, column, top_n=20):
    top_categories = df[column].value_counts().nlargest(top_n).index
    df[column] = df[column].apply(lambda x: x if x in top_categories else 'Other')
    return df


def evaluate_model(name, model, X_test, y_test, results):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name}:\n  RMSE: {rmse:.2f}\n  R²: {r2:.2f}\n")
    results[name] = {'RMSE': rmse, 'R2': r2}


def cross_validate_model(name, model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    mean_score = -scores.mean()
    std_score = scores.std()
    print(f"{name} CV RMSE: {mean_score:.2f} (+/- {std_score:.2f})\n")


def train_models(data_path):
    df = pd.read_csv(data_path)

    # Simplify high-cardinality categorical columns
    for col in ['neighbourhood', 'neighbourhood_group']:
        if col in df.columns:
            df = simplify_categories(df, col, top_n=20)


    # Drop high-cardinality text fields that aren’t useful
    df = df.drop(columns=['name', 'host_name'], errors='ignore')

    # One-hot encode the remaining categorical columns
    df = pd.get_dummies(df, drop_first=False)

    # Drop rows with missing values
    df = df.dropna()

    # Define features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    evaluate_model("Linear Regression", lr, X_test, y_test, results)
    cross_validate_model("Linear Regression", lr, X, y)
    models["linear_regression_model.pkl"] = lr

    # Random Forest (lower complexity to avoid memory error)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model("Random Forest", rf, X_test, y_test, results)
    cross_validate_model("Random Forest", rf, X, y)
    models["random_forest_model.pkl"] = rf

    # XGBoost (lower complexity)
    xgb = XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    evaluate_model("XGBoost", xgb, X_test, y_test, results)
    cross_validate_model("XGBoost", xgb, X, y)
    models["xgboost_model.pkl"] = xgb

    print("✅ All models trained and saved.")
    print(pd.DataFrame(results).T)
    
    # Save template columns used during training
    template_columns = list(X.columns)
    template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model_columns.pkl'))
    joblib.dump(template_columns, template_path)

    # Save models
    for filename, model in models.items():
        joblib.dump(model, os.path.join('..', 'models', filename))


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'engineered_features.csv'))
    train_models(data_path)
