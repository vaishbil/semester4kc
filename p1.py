import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load inbuilt dataset (Titanic)
df = sns.load_dataset("titanic")

# Identify missing values
print("Missing Values:\n", df.isnull().sum())

# Handle missing values:
# - Fill numerical columns with the median
# - Fill categorical columns with the mode
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Detect and remove outliers using IQR method
Q1 = df.select_dtypes(include=['number']).quantile(0.25)
Q3 = df.select_dtypes(include=['number']).quantile(0.75)
IQR = Q3 - Q1

df_cleaned = df[~((df.select_dtypes(include=['number']) < (Q1 - 1.5 * IQR)) |
                  (df.select_dtypes(include=['number']) > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nShape before outlier removal:", df.shape)
print("Shape after outlier removal:", df_cleaned.shape)