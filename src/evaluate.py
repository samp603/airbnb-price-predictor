import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load dataset
data_path = "../data/preprocessed_airbnb.csv"
df = pd.read_csv(data_path)

# Split features and target
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models
model_dir = "../models"
models = {
    "Linear Regression": joblib.load(os.path.join(model_dir, "linear_regression_model.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest_model.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgboost_model.pkl")),
}

results = {}

# Scatter plot of predictions vs. actual
plt.figure(figsize=(10, 6))
for name, model in models.items():
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "R²": r2}
    plt.scatter(y_test, preds, label=name, alpha=0.5)

plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.clf()

# Residuals for Random Forest
residuals = y_test - models["Random Forest"].predict(X_test)
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residuals Distribution - Random Forest")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("residuals_histogram.png")
plt.clf()

# RMSE bar plot
rmse_vals = [metrics["RMSE"] for metrics in results.values()]
sns.barplot(x=list(results.keys()), y=rmse_vals)
plt.title("Model RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("rmse_comparison.png")
plt.clf()

# Save results table
pd.DataFrame(results).T.to_csv("model_evaluation_results.csv")
