import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


input_path = "data/engineered_features.csv"
output_path = "data/preprocessed_airbnb.csv"


df = pd.read_csv(input_path)

#Separate features and target
if 'price' in df.columns:
    X = df.drop(columns=['price'])
    y = df['price']
else:
    X = df
    y = None

#Identify numeric features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Imputer
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features)
])

# Fit data
X_processed = preprocessor.fit_transform(X)

# Dataframe (create columns with features)
X_processed_df = pd.DataFrame(X_processed, columns=numeric_features)

# Add target column back
if y is not None:
    X_processed_df['price'] = y.values


os.makedirs(os.path.dirname(output_path), exist_ok=True)
X_processed_df.to_csv(output_path, index=False)

pipeline_path = output_path.replace('.csv', '_pipeline.pkl')
joblib.dump(preprocessor, pipeline_path)

print("✅ Preprocessing complete!")
print("Pipeline created!")


