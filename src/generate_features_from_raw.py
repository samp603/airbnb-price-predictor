import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib

# === Load Raw Data ===
raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'AB_NYC_2019.csv'))
df = pd.read_csv(raw_path)

# === Drop Useless or High Cardinality Columns ===
df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'], inplace=True, errors='ignore')

# === Filter Outliers ===
df = df[df['price'].between(20, 500)]
df = df[df['minimum_nights'] <= 30]

# === Add Clustered Location Feature ===
kmeans = KMeans(n_clusters=10, random_state=42)
df['location_cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])
df.drop(columns=['latitude', 'longitude'], inplace=True)

# Save KMeans model
kmeans_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'kmeans.pkl'))
os.makedirs(os.path.dirname(kmeans_path), exist_ok=True)
joblib.dump(kmeans, kmeans_path)

# === Add Engineered Features ===
df['reviews_per_month_per_year'] = df['reviews_per_month'] / 12

# Log transform skewed features
df['minimum_nights'] = np.log1p(df['minimum_nights'])
df['number_of_reviews'] = np.log1p(df['number_of_reviews'])
df['reviews_per_month'] = np.log1p(df['reviews_per_month'])
df['availability_365'] = np.log1p(df['availability_365'])

# Drop rows with missing
df.dropna(inplace=True)

# === Target ===
df['log_price'] = np.log1p(df['price'])

# === Normalize numeric ===
numeric_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
                'availability_365', 'reviews_per_month_per_year']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save Scaler
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'minmax_scaler.pkl'))
joblib.dump(scaler, scaler_path)

# === Save Processed Data ===
out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'engineered_features.csv'))
df.to_csv(out_path, index=False)
print(f"✅ Preprocessing complete: {out_path}")
