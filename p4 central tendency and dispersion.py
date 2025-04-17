import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
# Load inbuilt dataset (Titanic)
df = sns.load_dataset("titanic")
# Select numerical columns only
df_numeric = df.select_dtypes(include=['number'])

# Compute Measures of Central Tendency
mean_values = df_numeric.mean()
median_values = df_numeric.median()
mode_values = df_numeric.mode().iloc[0]  # Mode might return multiple values, take first row

# Compute Measures of Dispersion
variance_values = df_numeric.var()
std_dev_values = df_numeric.std()

# Display results
print("Mean Values:\n", mean_values)
print("\nMedian Values:\n", median_values)
print("\nMode Values:\n", mode_values)
print("\nVariance Values:\n", variance_values)
print("\nStandard Deviation Values:\n", std_dev_values)