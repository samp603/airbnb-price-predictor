from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/random_forest_model.pkl")  # Load best model

# Load encoder + scaler if used in pipeline
pipeline = joblib.load("../data/preprocessed_airbnb_pipeline.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_data = {
                'neighbourhood_group': request.form['neighbourhood_group'],
                'room_type': request.form['room_type'],
                'minimum_nights': float(request.form['minimum_nights']),
                'number_of_reviews': float(request.form['number_of_reviews']),
                'reviews_per_month': float(request.form['reviews_per_month']),
                'availability_365': float(request.form['availability_365'])
            }
            df = pd.DataFrame([input_data])
            processed = pipeline.transform(df)
            prediction = model.predict(processed)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)
