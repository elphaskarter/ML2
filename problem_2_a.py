import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
landis_data = np.load('landis_chlorophyl_regression.npy')  # Features (spectral bands)
landis_gt = np.load('landis_chlorophyl_regression_gt.npy')  # Target (chlorophyll content)

print(f"Shape of landis_data: {landis_data.shape}")
print(f"Data type of landis_data: {landis_data.dtype}")

band_names = ["Blue", "Green", "Yellow", "Orange", "Red 1", "Red 2", 
              "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1"]
bands = [490, 560, 600, 620, 650, 665, 705, 740, 842, 865]

# The DataFrame helps fro easier manipulation and EDA
df = pd.DataFrame(landis_data, columns=band_names)
df['Chlorophyll'] = landis_gt

# Data quality checks
print("Descriptive Statistics:") 
print(df.describe())

print("\n Checks for Missing Values")
print(df.isnull().sum())

# Feature Distributions preview (starts here)
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Spectral Bands and Chlorophyll")
plt.show()

# Multicollinearity analysis starts here
corr_matrix = df[band_names].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", xticklabels=band_names, yticklabels=band_names)
plt.title("Correlation Matrix of Spectral Bands")
plt.show()

# Variance Inflation Factor (VIF) per spectral band
vif_data = pd.DataFrame()
vif_data["Feature"] = band_names
vif_data["VIF"] = [variance_inflation_factor(df[band_names].values, i) for i in range(len(band_names))]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Pairwise Relationships
sns.pairplot(df[band_names])
plt.suptitle("Pairplot of Spectral Bands and Chlorophyll", y=1.02)
plt.show()

