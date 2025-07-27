
# ✅ Airbnb Price Predictor – Team TODOs (Complete Roadmap)

Welcome! This document lays out everything that still needs to be done for our project — broken down by category. Everything has context, suggestions, and space for you to claim tasks. Let’s collaborate and make this project amazing.

---

## 🎯 Project Goal Recap

We’re building a **machine learning tool to predict nightly Airbnb prices** based on features like location, room type, reviews, etc.

### Final goal:
- Users can input an Airbnb listing’s details (like neighborhood, room type, etc.) and get a **predicted price**.
- We’ll expose this as a simple **web app** using Flask (or Streamlit).
- All code, models, visualizations, and writeups will be hosted in our GitHub repo and GitHub Pages site.

---

## 🔄 Data Preprocessing

✅ **Clean raw data**  
> Done by Sam in `src/preprocess.py`. Loads the CSV, removes missing/outlier values, encodes categories, and saves cleaned data.

✅ **Feature Engineering Ideas**  
Try these in a notebook or new script:
- `price_per_night = price / minimum_nights`
- Log-transform highly skewed columns (`price`, `reviews_per_month`) to normalize distribution
- Normalize/standardize numeric columns

✅ **Use a Scikit-Learn Pipeline**  
Create a `preprocessing_pipeline.py` file to:
- Bundle steps (imputer, encoder, scaler) into a pipeline
- Easily reuse during modeling

---

## 📊 Exploratory Data Analysis (EDA)

✅ **Basic EDA started** (`notebooks/eda.ipynb`)

✅ **Add correlation heatmap**
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

✅ **Plot numeric distributions**  
Look at histograms or boxplots of:
- `number_of_reviews`
- `availability_365`
- `minimum_nights`

✅ **Analyze categorical features**
- Average price by `room_type`, `neighbourhood_group`
- Count plots (how many listings per category)

⬜ **Write `summary.md`**
Document 5–10 key findings from EDA — this helps everyone later during modeling.

---

## 🧠 Modeling

✅ **Baseline model (Linear Regression)**  
> Implemented by Sam in `train_model.py`

⬜ **Random Forest Regressor**
- Train `RandomForestRegressor(random_state=42)`
- Compare RMSE and R² to baseline
- Store results in a dictionary or table for comparison

⬜ **XGBoost Regressor**
- Train `xgboost.XGBRegressor()`
- Evaluate RMSE, R²
- Compare performance to previous models

⬜ **Hyperparameter Tuning**
> Optional but impactful
- Use `GridSearchCV` or `RandomizedSearchCV`
- Save best model parameters and performance

⬜ **Create `model_comparison.py`**
- Combine all model results into one table

---

## 📈 Evaluation & Visualization

⬜ **Prediction vs Actual Plot**

⬜ **Residuals Histogram**

⬜ **Bar Chart: Model RMSEs**

⬜ **Save Best Model to Disk**

---

## 🌐 Flask App (Stretch Goal)

⬜ **Build Simple Flask Web App**

⬜ **Deploy via GitHub Pages or Render**

---

## 📚 Documentation & Website

⬜ **README Enhancements**

⬜ **Team Member Contributions**

⬜ **Improve GitHub Pages Website**

⬜ **Final Presentation Slides**

---

## 🛠 Suggestions & Workflow Tips

- Use `git pull` before starting work
- Use `git add . && git commit -m "message"` to save changes
- Use `git push` to share
- Coordinate in group chat — avoid stepping on each other’s work

---

## 👥 Task Tracker

Make changes here as you work — or create issues/PRs in GitHub.

✅ = Done  
🟡 = In Progress  
⬜ = Not started
