from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import requests

app = Flask(__name__)

# Load model and preprocessing artifacts
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("data/minmax_scaler.pkl")
kmeans = joblib.load("models/kmeans.pkl")
template_columns = pd.read_csv("models/template_columns.csv").columns.tolist()

# === Address Geocoding ===
def geocode_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "AirbnbPricePredictor/1.0"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    else:
        raise ValueError("Could not geocode the provided address.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # User inputs
            min_nights = float(request.form["minimum_nights"])
            num_reviews = float(request.form["number_of_reviews"])
            reviews_per_month = float(request.form["reviews_per_month"])
            availability = float(request.form["availability_365"])
            neighbourhood_group = request.form["neighbourhood_group"]
            room_type = request.form["room_type"]
            address = request.form["address"]

            # Geocode address
            latitude, longitude = geocode_address(address)

            # Engineered features
            reviews_per_month_per_year = reviews_per_month / 12
            location_cluster = kmeans.predict(pd.DataFrame([[latitude, longitude]], columns=["latitude", "longitude"]))[0]

            # Log-transform numeric fields
            data = {
                "minimum_nights": np.log1p(min_nights),
                "number_of_reviews": np.log1p(num_reviews),
                "reviews_per_month": np.log1p(reviews_per_month),
                "availability_365": np.log1p(availability),
                "reviews_per_month_per_year": reviews_per_month_per_year,
                "location_cluster": location_cluster,
                "neighbourhood_group": neighbourhood_group,
                "room_type": room_type
            }

            df = pd.DataFrame([data])

            # One-hot encode categoricals
            df = pd.get_dummies(df, columns=["neighbourhood_group", "room_type"], prefix=["neighbourhood_group", "room_type"])

            # Normalize numeric features
            num_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                        'availability_365', 'reviews_per_month_per_year']
            df[num_cols] = scaler.transform(df[num_cols])

            # Add missing columns and align
            missing_cols = [col for col in template_columns if col not in df.columns]
            if missing_cols:
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
                df = pd.concat([df, missing_df], axis=1)
            df = df[template_columns]

            # Predict
            log_price = model.predict(df)[0]
            price = np.expm1(log_price)
            return render_template("index.html", prediction=f"${price:.2f}")

        except Exception as e:
            return render_template("index.html", prediction=f"❌ Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
