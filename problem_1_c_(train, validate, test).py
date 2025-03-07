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

    return data, labels

def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    return X_train, X_test, y_train, y_test

# XGBoost model training with early stopping and validation curves
def train_xgboost(X_train, y_train, X_val, y_val):

    # Convert data into DMatrix format (optimized for XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define parameters for XGBoost
    params = {
        'objective': 'multi:softmax',  # Multi-class classification
        'num_class': 9,               # Number of classes
        'eval_metric': 'mlogloss',    # Evaluation metric (log loss for multi-class)
        'max_depth': 6,               # Maximum depth of a tree
        'eta': 0.1,                   # Learning rate
        'subsample': 0.8,             # Subsample ratio of the training instances
        'colsample_bytree': 0.8,      # Subsample ratio of columns when constructing each tree
        'seed': 42                    # Random seed
    }

    # Train the model with early stopping
    evals_result = {}  # stores evaluation results
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=150,  # Maximum number of boosting rounds
        evals=[(dtrain, 'train'), (dval, 'val')],  # Evaluation sets
        early_stopping_rounds=15,  # Stop if no improvement for 10 rounds
        evals_result=evals_result,  # Store evaluation results
        verbose_eval=10  # Print evaluation results every 10 rounds
    )

    return model, evals_result

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    
    # Convert X_test to DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Predict using the model
    y_pred = model.predict(dtest)

    # Metrics computation
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

# Plot validation curves
def plot_validation_curves(evals_result):
    train_logloss = evals_result['train']['mlogloss']
    val_logloss = evals_result['val']['mlogloss']

    plt.figure(figsize=(12, 6))
    plt.plot(train_logloss, label='Training Log Loss')
    plt.plot(val_logloss, label='Validation Log Loss')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Log Loss')
    plt.title('Training and Validation Log Loss')
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Load data
    paviaU_data, gt_labels = load_data()
    paviaU_data = paviaU_data[:, :, :-1]

    # Preprocessing
    paviaU, paviaU_gt = preprocessing(paviaU_data, gt_labels)
    print(f"Data shape: {paviaU.shape}")
    print(f"Labels shape: {paviaU_gt.shape}")

    # Split data: train, test (70:30)
    X_train_, X_test, y_train, y_test = split_data(paviaU, paviaU_gt)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_)
    X_test  = scaler.transform(X_test)

    # split training data: train and validate
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Train XGBoost model without data balancing
    print("-" * 50)
    print("\n", "Without balancing the dataset")

    model, evals_result = train_xgboost(X_train, y_train, X_val, y_val)
    plot_validation_curves(evals_result)

    # Evaluate the model
    y_pred = model.predict(xgb.DMatrix(X_test))
    evaluate_model(model, X_test, y_test)

    # Train XGBoost model on Balanced data
    print("-" * 50)
    print("\n", "Balancing the dataset")

    # Balance the training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Balance the validation data
    X_val_balanced, y_val_balanced = balance_dataset(X_val, y_val)

    model_balanced, evals_result_balanced = train_xgboost(X_train_balanced, y_train_balanced, X_val_balanced, y_val_balanced)
    plot_validation_curves(evals_result_balanced)
    
    # Evaluate the model
    y_pred = model_balanced.predict(xgb.DMatrix(X_test))
    evaluate_model(model_balanced, X_test, y_test)