import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Load data
landis_data = np.load('landis_chlorophyl_regression.npy')  # Features (spectral bands)
landis_gt = np.load('landis_chlorophyl_regression_gt.npy') # Target (chlorophyll content)

band_names = ["Blue", "Green", "Yellow", "Orange", "Red 1", 
              "Red 2", "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1"]

df = pd.DataFrame(landis_data, columns=band_names)
df['Chlorophyll'] = landis_gt

# features (X) and target (y)
X = df[band_names].values
y = df['Chlorophyll'].values

# Splits the data into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# varying over layers with 10 neurons(feature data sets)
layer_results = {}

for num_layers in range(1, 6):

    # MLP layers
    hidden_layers = tuple([10] * num_layers)  # (10,), (10,10), (10,10,10), ...

    # Initialize the Multiple Layer Perceptron model
    mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                              activation='relu',
                              solver='adam',
                              max_iter=4500,
                              random_state=42)

    # Training the model
    mlp_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = mlp_model.predict(X_test)

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store regression Metrics
    layer_results[num_layers] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

    # Plot actual vs predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[0].set_title(f'Actual vs Predicted (Layers={num_layers})')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')

    # Plot residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='dashed')
    axes[1].set_title(f'Residual Plot (Layers={num_layers})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()

METRICS_DF = pd.DataFrame(layer_results).T
print("\nSummary of Results:")
print(METRICS_DF)

# Plot performance metrics
fig, ax = plt.subplots(figsize=(6, 4))
METRICS_DF[['MSE', 'RMSE', 'MAE']].plot(kind='bar', ax=ax)  
ax.set_title('MLP Performance for Different Hidden Layers')
ax.set_xlabel('Number of Layers')
ax.set_ylabel('Error Metrics')
ax.minorticks_on()
ax.tick_params(axis='both', which='both', direction='out', length=4) 
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend()
plt.show()