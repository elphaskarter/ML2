import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, r2_score


landis_data = np.load('landis_chlorophyl_regression.npy')    # Features (spectral bands)
landis_gt = np.load('landis_chlorophyl_regression_gt.npy')   # Target (chlorophyll content)

band_names = ["Blue", "Green", "Yellow", "Orange", "Red 1", 
              "Red 2", "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1"]

df = pd.DataFrame(landis_data, columns=band_names)
df['Chlorophyll'] = landis_gt

# Separating features (X) and target (y)
X = df[band_names].values
y = df['Chlorophyll'].values

# Train/Test (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing Feature Space
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Alternate the Number of Components from (1 to 10) a
component_range = range(1, 11)
training_r2_scores = []

for n_comp in component_range:
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train_scaled, y_train)
    
    # Predict on training set
    y_train_pred = pls.predict(X_train_scaled).ravel()
    
    # Compute R² on training data to determine pc
    r2_train = r2_score(y_train, y_train_pred)
    training_r2_scores.append(r2_train)

# Pick the best number of components based on highest training R²
best_n = component_range[np.argmax(training_r2_scores)]
best_r2 = max(training_r2_scores)
print(f"Best number of components from trained R²: {best_n}")
print(f"Training R² with {best_n} components: {best_r2:.3f}")

# Plot of training R² vs. number of components
plt.figure(figsize=(8, 5))
plt.plot(component_range, training_r2_scores, marker='o', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("Training R² Score")
plt.title("PLSR: Training R² vs. Number of Components")
plt.xticks(component_range)
plt.grid(True)
plt.show()

# Train PLSR Model with Best PC
pls_best = PLSRegression(n_components=best_n)
pls_best.fit(X_train_scaled, y_train)

# training and testing sets from best pc
y_train_pred = pls_best.predict(X_train_scaled).ravel()
y_test_pred = pls_best.predict(X_test_scaled).ravel()

# Computing the Metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    std_res = np.std(residuals)
    return mae, r2, std_res

mae_train, r2_train, std_train = compute_metrics(y_train, y_train_pred)
mae_test, r2_test, std_test = compute_metrics(y_test, y_test_pred)

print("\nFinal PLSR Model Performance:")
print(f"  Best Number of Components: {best_n}")

print("\nTraining Partition:")
print(f"  MAE: {mae_train:.3f}")
print(f"  R²: {r2_train:.3f}")
print(f"  Std. of Residuals: {std_train:.3f}")

print("\nTesting Partition:")
print(f"  MAE: {mae_test:.3f}")
print(f"  R²: {r2_test:.3f}")
print(f"  Std. of Residuals: {std_test:.3f}")

# Regression and Residual Plots
def plot_regression_and_residuals(y_true, y_pred, dataset_type="Training"):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 5))
    
    # Regression Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.xlabel("Actual Chlorophyll")
    plt.ylabel("Predicted Chlorophyll")
    plt.title(f"{dataset_type} Regression Plot\n"
              f"MAE: {mean_absolute_error(y_true, y_pred):.3f}, "
              f"R²: {r2_score(y_true, y_pred):.2f}")

    # Residual Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Chlorophyll")
    plt.ylabel("Residuals")
    plt.title(f"{dataset_type} Residual Plot\n"
              f"Std. Residuals: {np.std(residuals):.3f}")
    
    plt.tight_layout()
    plt.show()

# Plot for Training Data
plot_regression_and_residuals(y_train, y_train_pred, dataset_type="Training (PLSR)")

# Plot for Testing Data
plot_regression_and_residuals(y_test, y_test_pred, dataset_type="Testing (PLSR)")

 # summary metrics for the computed n_components
results = [] 
for n_comp in range(1, 11):
    # Create and fit PLSRegression model
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = pls.predict(X_train_scaled).ravel()
    y_test_pred  = pls.predict(X_test_scaled).ravel()
    
    # training and testing metrics
    mae_train, r2_train, std_train = compute_metrics(y_train, y_train_pred)
    mae_test, r2_test, std_test    = compute_metrics(y_test,  y_test_pred)
    
    # results
    results.append({'n_components': n_comp, 'train_MAE': mae_train, 'train_R2':  r2_train,
        'train_StdRes': std_train, 'test_MAE':  mae_test, 'test_R2':   r2_test, 'test_StdRes': std_test})

results_df = pd.DataFrame(results)
print("\nSummary of Training/Testing Metrics for Each Number of Components:")
print(results_df)

# report best n_components
best_idx = results_df['train_R2'].idxmax()
best_n   = results_df.loc[best_idx, 'n_components']
best_r2  = results_df.loc[best_idx, 'train_R2']
