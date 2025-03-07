# --Logistic regression--
# __Binary classification__
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from problem_1_a_i import rgb_image
from sklearn.metrics import (accuracy_score, precision_score, classification_report, 
                             confusion_matrix, recall_score, f1_score, roc_curve, roc_auc_score)
import io
import matplotlib
import contextlib

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from problem_1_a_i import data_cube

matplotlib.use(original_backend)
plt.close('all')

def compute_mean_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc

def class_stats(y_true, y_pred):
    classes = np.unique(y_true)
    per_class_accuracies = []
    
    for cls in classes:
        mask = y_true == cls
        cls_accuracy = accuracy_score(y_true[mask], y_pred[mask])
        per_class_accuracies.append(cls_accuracy)
    
    return np.mean(per_class_accuracies)

def preprocess_pavia(hsi_data, gt_labels, veg_classes):
    pavia_data = scipy.io.loadmat(hsi_data)
    pavia_gt = scipy.io.loadmat(gt_labels)

    hsi_data = pavia_data['paviaU']  
    hsi_data = hsi_data[:, :, :-1]
    gt_labels = pavia_gt['paviaU_gt']  

    n_counts = hsi_data.shape[0] * hsi_data.shape[1]  
    print(n_counts)

    X = hsi_data.reshape((n_counts, -1))  
    y = gt_labels.ravel()  # multiclass

    mask = y != 0  # Remove class 0 (if it exists)
    X = X[mask]
    y = y[mask] # multiclass mask

    # Define the binary mask of the labels (Vegetation: 2, 4)
    n_samples = np.where(np.isin(y, veg_classes), 1, 0)
    print(n_samples.shape)

    return gt_labels, X, y, n_samples

def plot(data, clsfd_img, gt_labels, veg_classes=[2, 4]):
    veg_mask = np.isin(clsfd_img, veg_classes)
    masked_veg = np.ma.masked_where(~veg_mask, clsfd_img)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot RGB image
    ax[0].imshow(data)
    ax[0].set_title('RGB Image', fontsize=8)
    ax[0].axis('on')

    # Plot Ground Truth
    tab10_cmap = plt.get_cmap("tab10", 10)  # 10 distinct colors
    class_names = {
        0: "unlabeled", 1: "asphalt", 2: "meadows", 3: "gravel",
        4: "trees", 5: "painted metal sheets", 6: "bare soil", 7: "bitumen",
        8: "self-blocking bricks", 9: "shadows"
    }
    
    im = ax[1].imshow(gt_labels, cmap=tab10_cmap, vmin=0, vmax=9)
    ax[1].set_title('Ground Truth', fontsize=8)
    ax[1].axis('on')
    
    cbar = fig.colorbar(im, ax=ax[1], ticks=range(10))
    cbar.set_ticklabels([class_names[i] for i in range(10)])

    # Plot Classified Vegetation
    ax[2].imshow(masked_veg, cmap=tab10_cmap)
    ax[2].set_title('Classified Vegetation', fontsize=8)
    ax[2].axis('on')
    
    plt.tight_layout()
    plt.show()

# Load the diabetes dataset
path_pavia_data = 'Pavia\PaviaU.mat'
path_pavia_gt = "Pavia\PaviaU_gt.mat"
gt_labels, X, y, bnry_mask_veg = preprocess_pavia(path_pavia_data, path_pavia_gt, [2, 4])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, bnry_mask_veg, test_size=0.3, stratify=y, random_state=42)

def get_band_index(wavelength, start=430, spacing=5):
    wave = int(round((wavelength - start) / spacing))
    return wave

blue_idx     = get_band_index(480)    
green_idx    = get_band_index(550)    
red_idx      = get_band_index(650)    
red_edge_idx = get_band_index(705)   
nir_idx      = get_band_index(800)    

idx = [blue_idx, green_idx, red_idx, red_edge_idx, nir_idx]
X_train_slctd = X_train[:, idx]
X_test_slctd  = X_test[:, idx]

# Further preprocessing
scaler = StandardScaler()
X_train_slctd = scaler.fit_transform(X_train_slctd)
X_test_slctd = scaler.transform(X_test_slctd)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42) 
model.fit(X_train_slctd, y_train)

# Evaluate the model
y_train_pred = model.predict(X_train_slctd)
y_test_pred  = model.predict(X_test_slctd)
y_test_probs = model.predict_proba(X_test_slctd)[:, 1]

# Evaluation metrics
metrics = {
    'Train Accuracy': accuracy_score(y_train, y_train_pred) * 100,
    'Test Accuracy': accuracy_score(y_test, y_test_pred) * 100,
    'Train Precision': precision_score(y_train, y_train_pred) * 100,
    'Test Precision': precision_score(y_test, y_test_pred) * 100,
    'Train Recall': recall_score(y_train, y_train_pred) * 100,
    'Test Recall': recall_score(y_test, y_test_pred) * 100,
    'Train F1': f1_score(y_train, y_train_pred) * 100,
    'Test F1': f1_score(y_test, y_test_pred) * 100
}

print("Performance Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.3f}%")

# Bais Metrics
bias = abs(metrics['Train Accuracy'] - metrics['Test Accuracy'])

# Variance Metrics
variance_f1 = abs(metrics['Train F1'] - metrics['Test F1'])
variance_recall = abs(metrics['Train Recall'] - metrics['Test Recall'])
variance_precision = abs(metrics['Train Precision'] - metrics['Test Precision'])

# Store results
bias_variance_metrics = {
    "Bias (Accuracy Gap)": bias,
    "Variance (F1-score difference)": variance_f1,
    "Variance (Precision difference)": variance_precision,
    "Variance (Recall difference)": variance_recall,
}

print("\nBias-Variance Metrics:")
for key, value in bias_variance_metrics.items():
    print(f"{key}: {value:.3f}")

# Compute mean accuracy per class
mean_accuracy = compute_mean_accuracy(y_train, y_train_pred)
mean_per_class_accuracy = class_stats(y_train, y_train_pred)

print(f"Mean Accuracy: {mean_accuracy:.3f}")
print(f"Mean Per-Class Accuracy:, {mean_per_class_accuracy:.3f}")


# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# ROC curve with Area Under the Curve (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)
auc_value = roc_auc_score(y_test, y_test_probs)

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]
print(f"Best threshold: {best_thresh: .2f}")

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_value:.2f}")
plt.xlabel("False Positive Rate", fontsize=8)
plt.ylabel("True Positive Rate", fontsize=8)
plt.title("ROC Curve for Vegetation Classification", fontsize=10)
plt.legend(loc="lower right", fontsize=8)
plt.show()

# Generate probabilities for the full dataset
X_full_slctd = X[:, idx]
X_full_slctd = scaler.transform(X_full_slctd)
y_full_probs = model.predict_proba(X_full_slctd)[:, 1]

# Convert probabilities to binary using best_threshold
y_full_pred_opt = (y_full_probs >= best_thresh ).astype(int)

# Remap binary predictions into a full image
full_pred_img = np.zeros(gt_labels.shape, dtype=int)
mask = (gt_labels != 0)
full_pred_img[mask] = y_full_pred_opt

# Remap '1' to '2' for vegetation and plot the predicted classes
full_pred_img_remap = full_pred_img.copy()
full_pred_img_remap[full_pred_img_remap == 1] = 2
plot(rgb_image, full_pred_img_remap, gt_labels, [2, 4])