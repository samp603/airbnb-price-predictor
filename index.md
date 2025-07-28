# 🏠 Airbnb Price Predictor

Welcome! This project is a machine learning-based tool that predicts nightly prices for Airbnb listings using real-world NYC data. It's designed to help new hosts estimate competitive prices in seconds.

👉 **[Try the live app here]([https://airbnb-price-predictor-csyh.onrender.com/])**  


---

## 🎯 Project Goals

- 🧠 Build a predictive ML model using open-source NYC Airbnb data  
- 📈 Provide hosts with a simple tool to estimate nightly listing prices  
- 🔍 Ensure transparency and explainability in every step of the pipeline  
- 🌐 Package it all into a clean, responsive Flask web application

---

## ⚙️ Tech Stack

- **Languages & Tools:** Python, Pandas, NumPy, scikit-learn, XGBoost, Flask  
- **Visualization & Analysis:** Seaborn, Matplotlib, Jupyter Notebooks  
- **Version Control:** GitHub  
- **Deployment:** Render  
- **Models:**  
  - Linear Regression (Baseline)  
  - Random Forest Regressor  
  - XGBoost Regressor

---

## 📁 Dataset

This project uses the publicly available **NYC Airbnb Open Data** from [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).  
It contains information on over 48,000 Airbnb listings including location, room type, availability, reviews, and more.

---

## 🚀 Current Features

- ✅ Cleaned, preprocessed, and feature-engineered dataset
- ✅ Multiple trained ML models with cross-validation
- ✅ Scaled numeric features and encoded categorical data
- ✅ Easy-to-use web interface (no latitude/longitude required!)
- ✅ Real-time price prediction via the Flask app

---

## 📊 ML Pipeline Overview

1. **Preprocessing:** Remove outliers, handle missing data, normalize fields
2. **Feature Engineering:** Add derived columns like `price_per_review`, `reviews_per_month_per_year`
3. **Modeling:** Train/test Linear Regression, Random Forest, and XGBoost
4. **Evaluation:** Compare RMSE and R² scores with cross-validation
5. **Deployment:** Package best model and deploy app via Flask on Render

---

## 🔧 Improvements Made

- Removed cheating features (like raw latitude/longitude) to avoid location leakage
- Replaced obscure input fields with user-friendly dropdowns and defaults
- Unified training and inference pipelines to prevent mismatch bugs
- Saved scaler and column templates for consistency across environments

---

## 📌 Next Steps

- 🔍 Add performance visualizations (residuals, predictions vs. actuals)
- 🎨 Improve styling/UI on the front-end
- 📦 Add Docker support (optional)
- 📘 Expand documentation and final project report
- 📊 Deploy interactive dashboards or plots with Streamlit

---

## 👥 Team Members

- **Sam Paris** – Lead Developer, Data Cleaning, Modeling, Web App  
- **Jeffrey** – Feature Engineering, Random Forest/XGBoost  
- **Elizabeth** – Exploratory Data Analysis, Visualizations

---

_This project is open source and maintained by the team.  
GitHub Pages built from `main` branch using `index.md`._

