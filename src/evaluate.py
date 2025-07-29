import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Setup Paths ===
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "../data/engineered_features.csv")
model_dir = os.path.join(BASE_DIR, "../models")
output_dir = os.path.join(BASE_DIR, "../evaluation_outputs")
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(data_path)
df = df[df["price"] <= 1000].dropna(subset=["log_price"])
df["price_dollars"] = np.expm1(df["log_price"])

# === Target and Features ===
X_raw = df.drop(columns=["price", "log_price", "price_dollars"], errors="ignore")
y = df["price_dollars"]

# === Load Preprocessing Pipeline ===
num_imputer = joblib.load(os.path.join(model_dir, "num_imputer.pkl"))
cat_imputer = joblib.load(os.path.join(model_dir, "cat_imputer.pkl"))
encoder = joblib.load(os.path.join(model_dir, "onehot_encoder.pkl"))

# === Preprocess ===
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()

X_num = pd.DataFrame(num_imputer.transform(X_raw[numeric_cols]), columns=numeric_cols)
X_cat_imputed = pd.DataFrame(cat_imputer.transform(X_raw[categorical_cols]), columns=categorical_cols)
X_cat_encoded = pd.DataFrame(
    encoder.transform(X_cat_imputed).toarray(),
    columns=encoder.get_feature_names_out(categorical_cols)
)

# Combine
X = pd.concat([X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)

# === Load Models ===
models = {
    "Linear Regression": joblib.load(os.path.join(model_dir, "linear_regression_model.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest_model.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
}

results = {}

# === Align Columns (handle missing or extra features) ===
trained_features = models["Linear Regression"].feature_names_in_
X = X.reindex(columns=trained_features, fill_value=0)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Plot 1: Actual vs Predicted ===
plt.figure(figsize=(12, 8))
colors = {
    "Linear Regression": "skyblue",
    "Random Forest": "orange",
    "XGBoost": "green"
}

for name, model in models.items():
    y_log_pred = model.predict(X_test)
    y_pred = np.expm1(y_log_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R²": r2}
    plt.scatter(y_test, y_pred, alpha=0.3, label=name, color=colors[name])

# Ideal line
min_val, max_val = y_test.min(), y_test.max()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Fit")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
plt.clf()

# === Plot 2: Residuals (Random Forest) ===
residuals = y_test - np.expm1(models["Random Forest"].predict(X_test))
sns.histplot(residuals, bins=50, kde=True, color='orange')
plt.title("Residuals Distribution - Random Forest")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_histogram.png"))
plt.clf()

# === Plot 3: RMSE Comparison ===
rmse_vals = [v["RMSE"] for v in results.values()]
sns.barplot(x=list(results.keys()), y=rmse_vals, palette=colors)
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rmse_comparison.png"))
plt.clf()

# === Save Evaluation Results ===
pd.DataFrame(results).T.to_csv(os.path.join(output_dir, "model_evaluation_results.csv"))

# Print sample output
print(df[["price", "log_price"]].sample(10))
print(f"✅ Evaluation complete. Files saved in: {output_dir}")
