# 🏠 Airbnb Price Predictor

Airbnb Price Predictor is a machine learning web application that estimates the nightly price of an Airbnb listing using real data from New York City. Built with scikit-learn, XGBoost, and Flask, this tool helps new hosts determine competitive prices based on listing characteristics like room type, host activity, and review metrics — without relying on GPS coordinates or other leaky features.

👉 [Try the Live App on Render](https://your-render-url.com)

---

## 📌 Project Summary

The goal of this project is to:

- Predict nightly Airbnb listing prices using NYC open data from Kaggle.
- Build and compare multiple machine learning models (Linear Regression, Random Forest, XGBoost).
- Deploy the best model in a user-friendly Flask web app.
- Ensure the model is explainable, reproducible, and free from data leakage.

The dataset contains thousands of real Airbnb listings with metadata like location (neighborhood), room type, host behavior, review history, and availability.

---

## ⚙️ Getting Started

Clone the repo and follow these steps to run everything locally:

### 1. Install Dependencies

```bash
pip install -r requirements.txt

2. Preprocess the Raw Dataset
python src/preprocess_data.py
This script:

Cleans and filters raw data

Caps outliers and fills missing values

Creates engineered features

One-hot encodes categorical variables

Saves a cleaned CSV to /data/engineered_features.csv

Saves a trained MinMaxScaler and a CSV column template for inference

3. Train the Models
python src/train_model.py
This script:

Trains three models: Linear Regression, Random Forest, and XGBoost

Evaluates RMSE and R² on test data

Saves models in the /models folder

Stores training column structure to prevent mismatches during prediction

🗂 Folder Structure
├── data/
│   ├── AB_NYC_2019.csv           ← Original dataset
│   ├── engineered_features.csv   ← Cleaned dataset with new features
│
├── models/
│   ├── minmax_scaler.pkl         ← Trained scaler for numeric features
│   ├── final_model_columns.pkl   ← Template for input columns
│   ├── xgboost_model.pkl         ← Final selected model
│
├── notebooks/
│   ├── eda.ipynb                 ← Visualizations, feature exploration
│
├── src/
│   ├── preprocess_data.py        ← Cleans and prepares data
│   ├── train_model.py            ← Trains and evaluates ML models
│   ├── app.py                    ← Flask app for predictions
│
├── templates/
│   └── index.html                ← Frontend of the web app
│
├── static/
│   └── style.css                 ← (Optional) Custom styles
│
├── requirements.txt
├── README.md
├── TODO.md
🌐 Live Web App
Our deployed web app allows users to input listing details like:

Room Type (Entire, Private, Shared)

Neighborhood Group (Brooklyn, Manhattan, etc.)

Host Listings Count

Reviews per Month

Availability per Year

Minimum Nights

And receive an estimated nightly price, all without requiring GPS coordinates.

📈 Models & Methods
Baseline: Linear Regression

Ensemble Models: Random Forest, XGBoost

Feature Engineering:

Price per review

Reviews per month per year

Is multi-listing host

Data Leakage Fixes:

Removed raw latitude/longitude

Dropped raw IDs and names

📎 Dataset
Source: New York City Airbnb Open Data on Kaggle

🚀 Deployment
The app is deployed on Render for free hosting of the Flask app.

To deploy yourself:

Push to a public GitHub repo

Create a new Web Service on Render

Select app.py as the entry point

Add a requirements.txt and optionally build.sh

📜 License
MIT License. Open to contributions.
