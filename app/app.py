from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to models and config
SCALER_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'minmax_scaler.pkl'))
COLUMNS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'template_columns.csv'))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'xgboost_model.pkl'))

# Load artifacts
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)
template_columns = pd.read_csv(COLUMNS_PATH).columns.tolist()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        user_input = {
            'minimum_nights': float(request.form['minimum_nights']),
            'number_of_reviews': float(request.form['number_of_reviews']),
            'reviews_per_month': float(request.form['reviews_per_month']),
            'calculated_host_listings_count': float(request.form['calculated_host_listings_count']),
            'availability_365': float(request.form['availability_365']),
            'neighbourhood_group': request.form['neighbourhood_group'],
            'room_type': request.form['room_type'],
        }

        # Derived features
        user_input['price_per_review'] = user_input['number_of_reviews'] / (user_input['number_of_reviews'] + 1)
        user_input['reviews_per_month_per_year'] = user_input['reviews_per_month'] / 12
        user_input['is_multi_listing_host'] = int(user_input['calculated_host_listings_count'] > 1)

        df = pd.DataFrame([user_input])

        # Scale numeric features
        numeric_cols = [
            'minimum_nights', 'number_of_reviews', 'reviews_per_month',
            'availability_365', 'reviews_per_month_per_year', 'price_per_review'
        ]
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # One-hot encode categorical values
        for col in ['neighbourhood_group', 'room_type']:
            col_value = f"{col}_{user_input[col]}"
            df[col_value] = 1
        df.drop(columns=['neighbourhood_group', 'room_type'], inplace=True)

        # Add missing columns
        for col in template_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[template_columns]

        # Predict
        price = model.predict(df)[0]
        return render_template('index.html', prediction=f"${price:.2f}")

    except Exception as e:
        return render_template('index.html', prediction=f"❌ Error: {str(e)}")

@app.route('/googleb572414e60f38549.html')
def google_verify():
    return app.send_static_file('googleb572414e60f38549.html')

if __name__ == '__main__':
    app.run(debug=True)
