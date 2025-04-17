
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import pandas as pd
import numpy as np

# Load inbuilt dataset (Diabetes dataset)
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Standardization (Z-score)
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)

# Normalization (Min-Max Scaling)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)

# Log Transformation (for skewed data)
df_transformed = df.copy()
df_transformed = df_transformed.apply(lambda x: np.log1p(x))

print("\nOriginal Data (first 5 rows):\n", df.head())
print("\nStandardized Data:\n", df_standardized.head())
print("\nNormalized Data:\n", df_normalized.head())
print("\nLog Transformed Data:\n", df_transformed.head())
