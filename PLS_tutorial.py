# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 21:42:02 2025

@author: elpha
"""

#Import necessary libraries
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Diabetes dataset
diabetes = datasets.load_diabetes()

# Extract features (X) and target variable (y)
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize PLS model with the desired number of components
n_components = 3
pls_model = PLSRegression(n_components=n_components)

# Fit the model on the training data
pls_model.fit(X_train, y_train)

# Predictions on the test set
y_pred = pls_model.predict(X_test)

# Evaluate the model performance
r_squared = pls_model.score(X_test, y_test)
print(f"R-Squared Error: {r_squared}")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize predicted vs actual values with different colors
plt.scatter(y_test, y_pred, c='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', c='red', label='Perfect Prediction')
plt.xlabel('Actual Diabetes Progression')
plt.ylabel('Predicted Diabetes Progression')
plt.title('PLS Regression: Predicted vs Actual Diabetes Progression')
plt.legend()
plt.show()

# Assuming 'pls_model' is your trained PLS regression model
# Assuming 'X_train' is your training set features
# Assuming 'X_test' is your test set features 

# # Create a SHAP explainer for the PLS regression model using KernelExplainer
# explainer = shap.KernelExplainer(pls_model.predict, X_train)

# # Calculate SHAP values for the entire test set
# shap_values = explainer.shap_values(X_test)

# # Summary plot for all instances
# shap.summary_plot(shap_values, X_test)

# Summarize the background data using k-means clustering
background_data = shap.kmeans(X_train, 10)  # Use 10 cluster centers as the background

# Create the KernelExplainer with the summarized background data
explainer = shap.KernelExplainer(pls_model.predict, background_data)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Summary plot for all instances
shap.summary_plot(shap_values, X_test)
