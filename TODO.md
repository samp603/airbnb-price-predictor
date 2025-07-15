# ✅ Airbnb Price Predictor – Team TODOs

Welcome! Here's a breakdown of remaining tasks with clear suggestions. Feel free to assign yourself to one and check it off as you go.

---

## 🔄 Data Preprocessing

- [x] **Clean raw data** (DONE by Sam in `preprocess_data.py`)
- [ ] **Explore feature engineering ideas**  
  _Suggestions_:  
  - Create a new feature: price per minimum nights (`price/minimum_nights`)  
  - Log-transform skewed columns like `price`, `reviews_per_month`  
  - Normalize numerical features (optional)

- [ ] **Add preprocessing pipeline to scikit-learn Pipeline object**  
  This makes the pipeline reusable and clean when training models.

---

## 📊 Exploratory Data Analysis (EDA)

- [x] Basic EDA notebook started (`notebooks/eda.ipynb`)
- [ ] **Add correlation heatmap**  
  Use `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')`

- [ ] **Visualize feature distributions**  
  Plot distributions of numerical features: `number_of_reviews`, `availability_365`, `minimum_nights`, etc.

- [ ] **Explore room_type and neighbourhood breakdown**  
  Bar plots for counts, avg prices, and maybe boxplots.

- [ ] **Create a short summary.md**  
  Note 5–10 key takeaways from EDA (trends, skew, potential features to drop/add)

---

## 🧠 Modeling

- [x] **Baseline Linear Regression** (DONE by Sam in `train_model.py`)
- [ ] **Implement Random Forest Regressor**  
  - Compare performance with baseline  
  - Use `RandomForestRegressor(random_state=42)`

- [ ] **Implement XGBoost Regressor**  
  - Use `xgboost.XGBRegressor()`  
  - Compare RMSE, R²

- [ ] **Hyperparameter tuning (bonus)**  
  Try `GridSearchCV` or `RandomizedSearchCV` for Random Forest and XGBoost

- [ ] **Create model_comparison.py**  
  A script that prints a table comparing model performance side-by-side

---

## 📈 Evaluation & Visualization

- [ ] **Prediction vs. Actual scatter plot**  
  Show how close the predictions are to true values

- [ ] **Residuals histogram**  
  Helps visualize prediction error

- [ ] **Bar chart of model RMSEs**  
  Visually compare model performance

- [ ] **Save best model to disk**  
  Use `joblib` or `pickle` so it can be reused for deployment

---

## 🌐 Stretch Goals

- [ ] **Build Streamlit or Flask web app**  
  Simple UI that lets users enter Airbnb listing info and get a predicted price

- [ ] **Deploy using GitHub Pages or Streamlit Cloud**

---

## 💡 Documentation / Presentation

- [ ] **Complete README sections**  
  Add usage instructions, model performance, credits, etc.

- [ ] **Finish `Team Member Contributions` in README**

- [ ] **Build and polish final project website**  
  Make sure it shows model outputs, graphs, and clean writeups

- [ ] **Make Presentation 

---

## 🙌 Team Roles (Flexible – can be changed)

| Name            | Main Contributions |
|-----------------|--------------------|
| Samuel Paris    | Repo setup, preprocessing, baseline model, README |
| Jeffrey Li      |  |
| Elizabeth Minor |  |

---

### 🔁 Suggestions

- Communicate progress and questions in the group chat
- Use `git pull` before working and `git push` after changes
- Use branches if working on big things
- Assign yourself in this file if you pick up a task

Let’s make this awesome!
