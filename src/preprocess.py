"""
preprocess.py - Data cleaning and preprocessing for Airbnb Price Predictor

This script loads the raw NYC Airbnb data and performs basic cleaning:
- Drops unnecessary or missing columns
- Caps outlier prices
- Encodes categorical variables
- Outputs a cleaned CSV for modeling

TODOs:
- Add more advanced feature engineering (e.g., price per night, room density)
- Try using Label Encoding instead of One-Hot to compare model impact
"""

import pandas as pd

# Load raw data
df = pd.read_csv('../data/AB_NYC_2019.csv')

# Drop rows with missing price or lat/lon
df.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)

# Drop columns not needed for modeling
df.drop(columns=['id', 'name', 'host_id',
        'host_name', 'last_review'], inplace=True)

# Fill missing reviews_per_month with 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Cap price outliers at $1000
df = df[df['price'] <= 1000]

# Encode categorical columns (one-hot encoding)
df = pd.get_dummies(df, columns=[
    'neighbourhood_group', 'neighbourhood', 'room_type'
], drop_first=True)

# Save cleaned data
df.to_csv('../data/cleaned_data.csv', index=False)
print("✅ Cleaned data saved to: ../data/cleaned_data.csv")
