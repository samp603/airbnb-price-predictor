import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


input_path="data/cleaned_data.csv"
output_path="data/engineered_features.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Input file not found: {input_path}")

df = pd.read_csv(input_path)

#Divide price per night by min nights (not sure if this is necessary)
# df["price_per_night"] = df["price"] / df["minimum_nights"]
# df["price_per_night"].replace([np.inf, -np.inf], np.nan, inplace=True)

#log skewed columns
for col in ["price", "reviews_per_month"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col]) 


#normalize numeric columns
numeric_cols = ["minimum_nights", "number_of_reviews", "availability_365", "reviews_per_month"]
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


#make "other" column to reduce amount of neighboorhoods
neighbourhood_cols = [col for col in df.columns if 'neighbourhood' in col.lower() or 'neighborhood' in col.lower()]

# Find the top 10 most common neighbourhoods by summing True values
neighbourhood_counts = {}
for col in neighbourhood_cols:
    if df[col].dtype == bool or df[col].isin([0, 1, True, False]).all():
        neighbourhood_counts[col] = df[col].sum()

# Get top 10 neighbourhoods
top_10_neighbourhoods = sorted(neighbourhood_counts.items(), key=lambda x: x[1], reverse=True)[:10]
top_neighbourhood_cols = [col for col, count in top_10_neighbourhoods]

# Keep only the top 10 neighbourhood columns and create an "Other" column
# First, create the "Other" column - True if none of the top 10 are True
df['neighbourhood_Other'] = ~df[top_neighbourhood_cols].any(axis=1)

# Keep only the top 10 neighbourhood columns plus the "Other" column
cols_to_keep = top_neighbourhood_cols + ['neighbourhood_Other']
cols_to_drop = [col for col in neighbourhood_cols if col not in top_neighbourhood_cols]

if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} less common neighbourhood columns")

print(f"Kept {len(cols_to_keep)} neighbourhood columns (top 10 + Other)")



os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Feature engineering complete. Saved to {output_path}")


