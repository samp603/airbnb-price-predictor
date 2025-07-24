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


# --- Step 1: Convert all boolean columns to 1s and 0s ---
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# --- Step 2: Identify neighborhood columns ---
neigh_cols = [col for col in df.columns if col.startswith("neighbourhood_")]
group_cols = [col for col in neigh_cols if col.startswith("neighbourhood_group_")]
spec_cols = [col for col in neigh_cols if col not in group_cols]

# --- Step 3: Convert neighborhood columns to 1s/0s (in case they were boolean too)
df[neigh_cols] = df[neigh_cols].astype(int)

# --- Step 4: Keep only top 10 specific neighborhoods
top_spec_cols = df[spec_cols].sum().sort_values(ascending=False).head(10).index.tolist()
other_spec_cols = [col for col in spec_cols if col not in top_spec_cols]

# --- Step 5: Create 'neighbourhood_other'
neigh_other = df[other_spec_cols].sum(axis=1).clip(upper=1).rename("neighbourhood_other")

# --- Step 6: Drop unused specific neighborhoods and add 'other'
df = df.drop(columns=[col for col in other_spec_cols if col in df.columns])
df = pd.concat([df, neigh_other], axis=1)
df = df.copy()  # Defragment the DataFrame

# --- Step 7: Optional column reordering ---
final_neigh_cols = group_cols + top_spec_cols + ["neighbourhood_other"]
df = df[[col for col in df.columns if col not in final_neigh_cols] + final_neigh_cols]




os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Feature engineering complete. Saved to {output_path}")


