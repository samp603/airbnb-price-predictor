import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
raw_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'AB_NYC_2019.csv'))
output_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'engineered_features.csv'))

# Load raw data
df = pd.read_csv(raw_path)

# Drop rows with missing critical values
df.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)

# Drop irrelevant or high-cardinality columns
df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'], inplace=True)

# Cap outliers
df = df[df['price'].between(20, 500)]
df = df[df['minimum_nights'] <= 30]

# Fill missing values
df['reviews_per_month'].fillna(0, inplace=True)

# Log-transform target and skewed variables
df['log_price'] = np.log1p(df['price'])
df['log_reviews_per_month'] = np.log1p(df['reviews_per_month'])

# Location clustering using KMeans
coords = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=20, random_state=42).fit(coords)
df['location_cluster'] = kmeans.labels_

# Drop raw coordinates to avoid leakage
df.drop(columns=['latitude', 'longitude'], inplace=True)

# Create derived non-leaky features
df['reviews_per_month_per_year'] = df['reviews_per_month'] / 12
df['is_multi_listing_host'] = (df['calculated_host_listings_count'] > 1).astype(int)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type', 'location_cluster'], drop_first=False)

# Normalize numeric features
scaler = MinMaxScaler()
numeric_cols = [
    'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'availability_365', 'reviews_per_month_per_year'
]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Drop raw price to avoid target leakage
df.drop(columns=['price'], inplace=True)

# Save the engineered dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Clean feature engineering complete. Saved to {output_path}")
