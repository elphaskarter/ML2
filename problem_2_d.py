import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

# Load data
landis_data = np.load('landis_chlorophyl_regression.npy')    # Features (spectral bands)
landis_gt = np.load('landis_chlorophyl_regression_gt.npy')   # Target (chlorophyll content)

band_names = ["Blue", "Green", "Yellow", "Orange", "Red 1", 
              "Red 2", "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1"]

df = pd.DataFrame(landis_data, columns=band_names)
df['Chlorophyll'] = landis_gt

# Separating features (X) and target (y)
X = df[band_names].values
y = df['Chlorophyll'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Design the MLP model with one hidden layer
mlp_model = MLPRegressor(hidden_layer_sizes=(10,),  # One hidden layer with 10 neurons
                   activation='relu',        # ReLU activation function
                   solver='adam',             # Adam optimizer
                   max_iter=5000,             # Maximum number of iterations
                   random_state=42
                #  solver='lbfgs'
                # learning_rate_init=0.001
                )

# Train the model
mlp_model.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R-squared: {r2:.3f}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Chlorophyll Content')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='dashed')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()