
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load inbuilt dataset (Penguins)
df = sns.load_dataset("penguins").dropna()  # Drop NaN values for visualization

# Scatter Plot: Relationship between Flipper Length and Body Mass
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species", style="sex")
plt.title("Scatter Plot: Flipper Length vs. Body Mass")
plt.show()

# Heatmap: Correlation between numerical features
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Numerical Features in Penguins Dataset")
plt.show()

# Boxplot: Detect outliers in Bill Length by Species
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="species", y="bill_length_mm", hue="sex")
plt.title("Boxplot: Bill Length by Species")
plt.show()