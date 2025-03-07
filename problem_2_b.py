# --
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
landis_data = np.load('landis_chlorophyl_regression.npy')  
landis_gt = np.load('landis_chlorophyl_regression_gt.npy')
band_names = ["Blue", "Green", "Yellow", "Orange", "Red 1", "Red 2", "Red Edge 1", 
              "Red Edge 2", "NIR_Broad", "NIR1"]

df = pd.DataFrame(landis_data, columns=band_names)
df['Chlorophyll'] = landis_gt

# Preprocessing
X = df[band_names].values
y = df['Chlorophyll'].values

# Training and testing sets (70% training and 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize band data (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on training and testing sets
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Compute performance metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    std_res = np.std(residuals)
    return mae, r2, std_res

mae_train, r2_train, std_train = compute_metrics(y_train, y_train_pred)
mae_test, r2_test, std_test = compute_metrics(y_test, y_test_pred)

print("Training Metrics:")
print(f"MAE: {mae_train:.3f}")
print(f"R-squared: {r2_train:.3f}")
print(f"Std. of Residuals: {std_train:.3f}")

print("\nTesting Metrics:")
print(f"MAE: {mae_test:.3f}")
print(f"R-squared: {r2_test:.3f}")
print(f"Std. of Residuals: {std_test:.3f}")

# Create regression and residual plots
def plot_regression_and_residuals(y_true, y_pred, dataset_type="Training"):
    residuals = y_true - y_pred
    
    # Regression plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.xlabel("Actual Chlorophyll")
    plt.ylabel("Predicted Chlorophyll")
    plt.title(f"{dataset_type} Regression Plot\nMAE: {mean_absolute_error(y_true, y_pred):.2f}, R²: {r2_score(y_true, y_pred):.2f}")
    
    # Residual plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Chlorophyll")
    plt.ylabel("Residuals")
    plt.title(f"{dataset_type} Residual Plot\nStd. Residuals: {np.std(residuals):.2f}")
    
    plt.tight_layout()
    plt.show()

# Plot for training data
plot_regression_and_residuals(y_train, y_train_pred, dataset_type="Training")

# Plot for testing data
plot_regression_and_residuals(y_test, y_test_pred, dataset_type="Testing")
