✅ Airbnb Price Predictor – Team TODOs (Updated Roadmap)
Welcome! This doc tracks everything we’ve done (and still need to do) for our Airbnb price prediction project. It’s broken down by category, with clear context and actionables. We’ve made great progress — let’s finish strong.

🎯 Project Goal Recap
We’re building a machine learning tool to predict nightly Airbnb prices using host and listing details like location, room type, and reviews.

Final deliverables:
A clean, user-friendly Flask web app where users input listing info and receive a predicted price

Fully-trained and validated ML models (Linear Regression, Random Forest, XGBoost)

All code, models, and docs stored in GitHub, with a polished project page via GitHub Pages

🔄 Data Preprocessing
✅ Cleaned Raw Data

preprocess_features.py now fully cleans the original dataset (removing missing values, capping outliers, etc.)

✅ Removed Cheating Features

We removed latitude and longitude to prevent location leakage (aka model cheating)

✅ New Engineered Features

price_per_review

reviews_per_month_per_year

is_multi_listing_host

Normalized all numeric fields using MinMaxScaler

✅ Saved Scaler + Template Columns

Now reusable in both training and the Flask app

✅ Dropped Irrelevant Fields

Removed high-cardinality or useless columns (name, host_name, neighbourhood, etc.)

📊 Exploratory Data Analysis (EDA)
✅ Initial EDA (notebooks/eda.ipynb)

Explored value counts, missing data, and price distributions

✅ Correlation Heatmap & Distribution Plots

Identified skew and correlation across numeric features

🟡 Write summary.md

Brief summary of 5–10 insights from EDA — helps justify modeling decisions later

🧠 Modeling
✅ Baseline (Linear Regression)

Already trained and evaluated

✅ Random Forest & XGBoost

Both trained with simplified complexity (to avoid memory issues and overfitting)

✅ Cross-Validation Added

All models are validated with 5-fold CV for better generalization comparison

✅ Model Artifacts Saved

Models saved as .pkl, including final column template and scaler

🟡 Hyperparameter Tuning (Optional)

Could still improve XGBoost and RF further

🟡 Create model_comparison.py

Compare RMSE and R² for all models in one summary table or plot

📈 Evaluation & Visualization
✅ Prediction vs Actual Plot
✅ Residuals Histogram
✅ Bar Chart: Model RMSEs
✅ Print Evaluation in train_model.py

🌐 Flask Web App
✅ Clean and Functional UI

Removed confusing fields like latitude/longitude — users now input understandable values only

✅ Predicts Using Real Trained Model

Web app uses the exact same scaler and features as training

🟡 Improve Form Styling / UX

Current layout works but could be more visually polished

🟡 Deploy to Render or GitHub Pages

Currently runs locally — next step is deploying online

📚 Documentation & Presentation
🟡 Update README.md with full project context

🟡 Team Contributions Section

List who did what (EDA, modeling, UI, etc.)

🟡 Update GitHub Pages Site

Add demo screenshots, instructions, and project summary

⬜ Final Presentation Slides
