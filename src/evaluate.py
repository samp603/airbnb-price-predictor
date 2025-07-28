import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# === Setup ===
# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/engineered_features.csv")
df = pd.read_csv(data_path)

# Cap prices to reduce outlier influence
df = df[df["price"] <= 1000]

# Split features and target
X = df.drop(columns=["price"])
y = df["price"]

# One-hot encode to match training format
X = pd.get_dummies(X, drop_first=True)

# Load a sample model to get the correct feature order
model_dir = os.path.join(os.path.dirname(__file__), "../models")
sample_model = joblib.load(os.path.join(model_dir, "linear_regression_model.pkl"))
trained_features = sample_model.feature_names_in_

# Align features with training time (handle missing or extra columns)
X = X.reindex(columns=trained_features, fill_value=0)

# Train/test split
X_train, X_test, y_yetrain, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load all models
models = {
    "Linear Regression": sample_model,
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest_model.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgboost_model.pkl")),
}

# Output directory
output_dir = os.path.join(os.path.dirname(__file__), "../evaluation_outputs")
os.makedirs(output_dir, exist_ok=True)

results = {}

# === Plot 1: Actual vs. Predicted Scatter ===
plt.figure(figsize=(12, 8))
colors = {
    "Linear Regression": "skyblue",
    "Random Forest": "orange",
    "XGBoost": "green"
}

for name, model in models.items():
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "R²": r2}
    plt.scatter(y_test, preds, alpha=0.3, label=name, color=colors[name])

# Ideal fit line
min_val, max_val = y_test.min(), y_test.max()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal Fit")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
plt.clf()

# === Plot 2: Residuals for Random Forest ===
residuals = y_test - models["Random Forest"].predict(X_test)
sns.histplot(residuals, bins=50, kde=True, color='orange')
plt.title("Residuals Distribution - Random Forest")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_histogram.png"))
plt.clf()

# === Plot 3: RMSE Comparison ===
rmse_vals = [metrics["RMSE"] for metrics in results.values()]
sns.barplot(x=list(results.keys()), y=rmse_vals, palette=colors)
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rmse_comparison.png"))
plt.clf()

# === Save Results Table ===
pd.DataFrame(results).T.to_csv(os.path.join(output_dir, "model_evaluation_results.csv"))

print(f"✅ Evaluation complete. Files saved in: {output_dir}")
