import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Paths
BASE_DIR = os.path.dirname(__file__)
raw_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'AB_NYC_2019.csv'))
output_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'engineered_features.csv'))
scaler_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'minmax_scaler.pkl'))
template_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'models', 'template_columns.csv'))

# Load raw data
df = pd.read_csv(raw_path)

# Drop rows with missing values in critical columns
df.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)
df['reviews_per_month'].fillna(0, inplace=True)

# Cap outliers
df = df[df['price'].between(20, 500)]
df = df[df['minimum_nights'] <= 30]

# Derived features
df['price_per_review'] = df['price'] / (df['number_of_reviews'] + 1)
df['reviews_per_month_per_year'] = df['reviews_per_month'] / 12
df['is_multi_listing_host'] = (df['calculated_host_listings_count'] > 1).astype(int)

# Normalize numeric features
scaler = MinMaxScaler()
numeric_cols = [
    'minimum_nights', 'number_of_reviews', 'reviews_per_month',
    'availability_365', 'reviews_per_month_per_year', 'price_per_review'
]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Drop high-cardinality or irrelevant columns
df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood'], inplace=True, errors='ignore')

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], drop_first=False)

# Save scaler
joblib.dump(scaler, scaler_path)

# Save template column structure (excluding target variable)
df_features = df.drop(columns=['price'], errors='ignore')

# Convert all bool columns (one-hot encoded) to int (0/1)
for col in df_features.select_dtypes(include='bool').columns:
    df_features[col] = df_features[col].astype(int)

# Save one example row with correct data types
df_features.iloc[0:1].to_csv(template_path, index=False)


# Save cleaned dataset
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved to {output_path}")
print(f"📐 Template columns saved to {template_path}")
print(f"🧪 Scaler saved to {scaler_path}")
