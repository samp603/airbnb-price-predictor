import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def train_models(data_path):
    df = pd.read_csv(data_path)
    df = df[df['price'] <= 1000]
    df.dropna(subset=['log_price'], inplace=True)
    df['price_dollars'] = np.expm1(df['log_price'])

    X_raw = df.drop(columns=['price', 'log_price', 'price_dollars'], errors='ignore')
    y_log = df['log_price']
    y_true_price = df['price_dollars']

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute numerics
    num_imputer = SimpleImputer(strategy='mean')
    X_num = pd.DataFrame(num_imputer.fit_transform(X_raw[numeric_cols]), columns=numeric_cols)

    # Impute + encode categoricals
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_raw[categorical_cols]), columns=categorical_cols)

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_cat_encoded = pd.DataFrame(
        encoder.fit_transform(X_cat_imputed).toarray(),
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Combine all features
    X_final = pd.concat([X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)
    
    # Define model_dir early so it's available below
    base_dir = os.path.dirname(data_path)
    model_dir = os.path.join(base_dir, "../models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save column template for inference use
    template_path = os.path.join(model_dir, "template_columns.csv")
    pd.DataFrame(columns=X_final.columns).to_csv(template_path, index=False)
    
    # === Save preprocessing artifacts ===
    joblib.dump(num_imputer, os.path.join(model_dir, "num_imputer.pkl"))
    joblib.dump(cat_imputer, os.path.join(model_dir, "cat_imputer.pkl"))
    joblib.dump(encoder, os.path.join(model_dir, "onehot_encoder.pkl"))

    # === Train/Test split ===
    X_train, X_test, y_log_train, y_log_test, y_true_train, y_true_test = train_test_split(
        X_final, y_log, y_true_price, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        tree_method='hist',
        n_jobs=-1
    )

    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_log_train)
        y_log_pred = model.predict(X_test)
        y_pred = np.expm1(y_log_pred)

        rmse_val = rmse(y_true_test, y_pred)
        r2_val = r2_score(y_true_test, y_pred)
        results[name] = {'RMSE': rmse_val, 'R2': r2_val}

        # Cross-Validation on log-price
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        log_cv_scores = cross_val_score(model, X_final, y_log, scoring='neg_root_mean_squared_error', cv=cv)
        log_cv_rmse = -log_cv_scores.mean()
        log_cv_std = log_cv_scores.std()

        print(f"{name}:")
        print(f"  RMSE: {rmse_val:.2f}")
        print(f"  R²: {r2_val:.2f}")
        print(f"  Log-Price CV RMSE: {log_cv_rmse:.4f} (+/- {log_cv_std:.4f})\n")

        # Save model
        model_path = os.path.join(model_dir, f"{name.replace(' ', '_').lower()}_model.pkl")
        joblib.dump(model, model_path)

    print("✅ All models trained and saved.")
    print(pd.DataFrame(results).T)

if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'engineered_features.csv'))
    train_models(data_path)
