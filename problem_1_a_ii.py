# --pca analysis--
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from problem_1_a_i import data_cube, wavelengths, points, material
import contextlib
import io
import matplotlib

original_backend = matplotlib.get_backend()  
matplotlib.use('Agg')  

with contextlib.redirect_stdout(io.StringIO()):
    from problem_1_a_i import data_cube, wavelengths, points, material

matplotlib.use(original_backend)
plt.close('all')

data_cube = data_cube[:, :, :-1]
rows, cols, bands = data_cube.shape
X = data_cube.reshape((rows * cols, bands))
wavelengths = wavelengths[:-1]

# covariance matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cov_matrix = np.cov(X_scaled, rowvar=False)
print("Covariance matrix shape:", cov_matrix.shape)

# Perform PCA for all features
pca = PCA()
X_pca = pca.fit_transform(X) 

# # Perform PCA
# pca_10 = PCA(n_components=10)
# pca_10.fit(X)

# # Ppc scores onto 10 components
# X_score_10 = pca_10.transform(X) 
# pca_cube = X_score_10.reshape((rows, cols, 10))

# # Reconstructed data in 10 pc dimension
# rcnstrd_data = pca_10.inverse_transform(X_score_10)  
# rcnstrd_data = rcnstrd_data.reshape((rows, cols, bands))

# 99% variance
var_ratios = pca.explained_variance_ratio_
cum_var = np.cumsum(var_ratios)
n_99 = np.argmax(cum_var >= 0.99) + 1

print(f'PCs explainig 99% variance: {n_99}')

X_pca_denoised = X_pca.copy()
X_pca_denoised[:, n_99:] = 0
X_rcnstrctd = pca.inverse_transform(X_pca_denoised)
denoised_data = X_rcnstrctd.reshape((rows, cols, bands))

# Plot each principal component
n_components = n_99
n_cols = 4
n_rows = (n_components + n_cols - 1) // n_cols   

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
axes = axes.ravel()
for i in range(n_components):
    axes[i].imshow(denoised_data[:, :, i], cmap=cmocean.cm.thermal)
    axes[i].set_title(f"PCs Explaining 99% Variance:\n PC {i+1}")
    axes[i].axis("off")
for i in range(n_components, n_rows * n_cols):
    axes[i].axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cov_matrix, cmap='inferno', interpolation='nearest', aspect='auto')
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix')
plt.xlabel('Bands')
plt.ylabel('Bands')
plt.show()

# Create a figure with subplots for denoised data
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()  
for i, (x, y) in enumerate(points):
    ax = axes[i]
    x_int = int(round(x))  # Round to the nearest integer
    y_int = int(round(y))  # Round to the nearest integer
    spctrm = denoised_data[x_int, y_int, :]  # Extract the spectrum at (x_int, y_int)
    spctrm_nrmlzd = (spctrm - np.min(spctrm)) / (np.max(spctrm) - np.min(spctrm))
    ax.plot(wavelengths, spctrm_nrmlzd, color='black')
    ax.set_title(f'{material[i]}')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')

plt.tight_layout()
plt.show()

def calculate_snr(image, no_data_value=np.nan):
    snr = np.zeros(image.shape[2])  
    for band in range(image.shape[2]):
        band_data = image[:, :, band].flatten()
        if np.isnan(no_data_value):
            valid_pixels = band_data[~np.isnan(band_data)]
        else:
            valid_pixels = band_data[band_data != no_data_value]
        mu = np.mean(valid_pixels)
        sigma = np.std(valid_pixels)
        snr[band] = mu / sigma if sigma != 0 else np.nan
    return snr

# Calculate SNR on original scale
snr_original = calculate_snr(data_cube)
snr_denoised = calculate_snr(denoised_data)

print("SNR (Original Image):", np.nanmean(snr_original))
print("SNR (Transformed Image):", np.nanmean(snr_denoised))

# Plot SNR vs. wavelength
fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
if wavelengths is not None and len(wavelengths) == bands:
    x_axis = wavelengths
    x_label = "Wavelength (nm)"
else:
    x_axis = np.arange(bands)
    x_label = "Band Index"

# Plot SNR for original data (all PCs)
axs[0].plot(x_axis, snr_original, label="All PCs")
axs[0].set_xlabel(x_label)
axs[0].set_ylabel("SNR (Mean / Std)")
axs[0].set_title("SNR - All PCs")
axs[0].grid(True)
axs[0].legend()

# Plot SNR for denoised data (99% PCA)
axs[1].plot(x_axis, snr_denoised, color='orange', label="denoised (99%)")
axs[1].set_xlabel(x_label)
axs[1].set_ylabel("SNR (Mean / Std)")  # Keep y-label for clarity
axs[1].set_title("SNR - 99% PCA")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

def error(original, reconstr):
    return np.mean((original - reconstr) ** 2)

max_components = X.shape[1]
errors = []
pc_range = range(1, max_components + 1)

for k in pc_range:
    pca_k = PCA(n_components=k)
    X_pca_k = pca_k.fit_transform(X)
    X_reconstr_k = pca_k.inverse_transform(X_pca_k)
    err = error(X, X_reconstr_k)
    errors.append(err)

plt.figure(figsize=(8, 6))
plt.plot(pc_range, errors, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Mean Squared Error (Reconstruction)")
plt.show()