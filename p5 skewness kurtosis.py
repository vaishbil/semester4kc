
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load inbuilt dataset (Penguins)
df = sns.load_dataset("penguins")
# Select numerical columns only
df_numeric = df.select_dtypes(include=['number']).dropna()  # Drop NaN values for calculations

# Plot histograms
df_numeric.hist(bins=20, figsize=(10, 6), edgecolor='black', grid=False)
plt.suptitle("Histograms of Numerical Features in Penguins Dataset", fontsize=14)
plt.show()

# Compute Skewness
skewness_values = df_numeric.skew()
print("\nSkewness Values:\n", skewness_values)

# Compute Kurtosis
kurtosis_values = df_numeric.kurtosis()
print("\nKurtosis Values:\n", kurtosis_values)
