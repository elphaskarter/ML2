# XGBOOST multicalss classification
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

def get_band_index(wavelength, start=430, spacing=5):
    wave = int(round((wavelength - start) / spacing))
    return wave

def load_data():
    data = scipy.io.loadmat('Pavia\PaviaU.mat')['paviaU']
    labels = scipy.io.loadmat('Pavia\PaviaU_gt.mat')['paviaU_gt']
    return data, labels

def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def preprocessing(data, labels):
    mask = labels != 0  # non-zero labels mask
    data = data[mask]   
    labels = labels[mask]  

    # Flatten the data and labels
    data = data.reshape(data.shape[0], -1)  # Flatten the data
    labels = labels.flatten()  # Flatten the labels

    # shifting class labels to start at 1
    labels = labels - 1

    # Get indices for selected bands
    blue_idx     = get_band_index(480)     # Blue band
    green_idx    = get_band_index(550)     # Green band
    red_idx      = get_band_index(650)     # Red band
    red_edge_idx = get_band_index(705)     # Red-edge band
    nir_idx      = get_band_index(800)     # Near-infrared band

    # Select the bands
    selected_bands = [blue_idx, green_idx, red_idx, red_edge_idx, nir_idx]
    data = data[:, selected_bands]

    return data, labels, selected_bands

def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    return X_train, X_test, y_train, y_test

# XGBoost model training (No Hyperparameter tuning)
def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=9, random_state=42)
    # model = xgb.XGBClassifier(objective='multi:softprob', num_class=9, random_state=42)
    model.fit(X_train, y_train) # Features(X), Labels(Y)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # metrics computation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Model's performance
    print(f"Mean Accuracy: {accuracy:.4f}")
    print(f"Mean Per-Class Precision: {precision:.4f}")
    print(f"Mean Per-Class Recall: {recall:.4f}")
    print(f"Mean Per-Class F1-Score: {f1:.4f}")

    # Confusion matrix and plotting
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(9), yticklabels=np.arange(9))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Feature importance plotting
    fig, ax = plt.subplots(figsize=(6,4))
    xgb.plot_importance(model, ax=ax, importance_type='gain', show_values=True) 
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for text_obj in ax.texts:
        original_text = text_obj.get_text()
        try:
            val = float(original_text)
            text_obj.set_text(f"{val:.3f}")
        except ValueError:
            pass

    ax.set_title("Feature Importance (Gain)")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Features (Bands)")
    ax.grid(False)

    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    # Load data
    paviaU_data, gt_labels = load_data()
    paviaU_data = paviaU_data[:, :, :-1]

    # Preprocessing
    paviaU, paviaU_gt, bands = preprocessing(paviaU_data, gt_labels)
    print(f"Data shape: {paviaU.shape}")
    print(f"Labels shape: {paviaU_gt.shape}")

    # Split data (70:30)
    X_train, X_test, y_train, y_test = split_data(paviaU, paviaU_gt)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("-" * 50)
    print("\n", "Without balancing the dataset")

    # Train XGBoost model without data balancing
    model = train_xgboost(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print("-" * 50)
    print("\n", "Balancing the dataset")

    # Train XGBoost model on Balnaced data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    model_balanced = train_xgboost(X_train_balanced, y_train_balanced)
    evaluate_model(model_balanced, X_test, y_test)

    # distribution before SMOTE balancing
    unique, counts = np.unique(y_train, return_counts=True)
    print(f'Distribution before SMOTE balancing:\n {counts}')  

    # distribution after SMOTE balancing
    unique_bal, counts_bal = np.unique(y_train_balanced, return_counts=True)
    print(f'Distribution after SMOTE balancing:\n {counts_bal}')  

    # predictions based on the objective used (testing)
    y_pred = model.predict(X_test) # objective=multi:softmax
    # y_proba = model.predict_proba(X_test)  # objetctive=multi:softprob
    # y_pred_1d = np.argmax(y_pred, axis=0)
    
    # Classification report
    print(f'Classification report:\n {classification_report(y_test, y_pred)}')

    height, width, num_bands = paviaU_data.shape  # (610, 340, 103)
    X_full = paviaU_data.reshape(height*width, num_bands)  # (610*340, 103)
    X_full_selected = X_full[:, bands]
    X_full_scaled = scaler.transform(X_full_selected)
    y_full_pred_1d = model.predict(X_full_scaled) 
    y_full_pred_2d = y_full_pred_1d.reshape((height, width))

    # Plot
    plt.figure(figsize=(6,6))
    plt.imshow(y_full_pred_2d, cmap='tab10')
    plt.title("Predicted Classes", fontsize=8)
    plt.axis('on')
    plt.show()

    # # Sentinel data usage
    # class_key = {
        # 0: "unlabeled",
        # 1: "asphalt",
        # 2: "meadows",
        # 3: "gravel",
        # 4: "trees",
        # 5: "painted metal sheets",
        # 6: "bare soil",
        # 7: "bitumen",
        # 8: "self-blocking bricks",
        # 9: "shadows"
    # } 

    # predicted_labels = model.predict(sentinel_data_flat)  # Predicted labels (e.g., [1, 2, 3, ...])
    # predicted_classes = [class_key[label] for label in predicted_labels]  # Convert to class names
    # Reshape predictions into a classification map
    # classification_map = np.array(predicted_classes).reshape(sentinel_data.shape[0], sentinel_data.shape[1])

    # # Visualize the classification map
    # import matplotlib.pyplot as plt
    # plt.imshow(classification_map, cmap="viridis")
    # plt.colorbar()
    # plt.show()

    # Make predictions on Sentinel-2 data
    # class_names = {
    #     0: "unlabeled",
    #     1: "asphalt",
    #     2: "meadows",
    #     3: "gravel",
    #     4: "trees",
    #     5: "painted metal sheets",
    #     6: "bare soil",
    #     7: "bitumen",
    #     8: "self-blocking bricks",
    #     9: "shadows"
    # }

    # predicted_labels = model.predict(sentinel_data_flat)  # Predicted labels (e.g., [1, 2, 3, ...])

    # # Reshape predictions into a classification map
    # classification_map = predicted_labels.reshape(sentinel_data.shape[0], sentinel_data.shape[1])

    # # Define the `tab10` colormap
    # tab10_cmap = plt.get_cmap("tab10", 10)  # 10 distinct colors

    # # Define class names
    # class_names = {
    #     0: "unlabeled",
    #     1: "asphalt",
    #     2: "meadows",
    #     3: "gravel",
    #     4: "trees",
    #     5: "painted metal sheets",
    #     6: "bare soil",
    #     7: "bitumen",
    #     8: "self-blocking bricks",
    #     9: "shadows"
    # }
    # # Plot the classification map
    # plt.figure(figsize=(10, 10))
    # plt.imshow(classification_map, cmap=tab10_cmap)

    # # Add a colorbar with class names
    # cbar = plt.colorbar(ticks=range(10))
    # cbar.ax.set_yticklabels([class_names[i] for i in range(10)])  # Set class names as tick labels

    # plt.title("Classification Map (tab10 colormap)")
    # plt.show()

    # # Plot the classification map
    # plt.figure(figsize=(10, 10))
    # plt.imshow(classification_map, cmap=tab10_cmap)
    # plt.colorbar(ticks=range(10), label="Class Labels")
    # plt.title("Classification Map (tab10 colormap)")
    # plt.show()
