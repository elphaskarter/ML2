# --Data Interpretation--
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def stretch(band, stretch_type='minmax', percentiles=(2, 98)):
    band_clean = band[~np.isnan(band)]

    if stretch_type == 'percentile':# Percentile stretching
        p_low, p_high = np.nanpercentile(band_clean, percentiles) 
        band = np.clip(band, p_low, p_high)
        return (band - p_low) / (p_high - p_low)
    
    elif stretch_type == 'minmax': # Classic min-max stretching
        bmin = np.nanmin(band_clean)
        bmax = np.nanmax(band_clean)
        return (band - bmin) / (bmax - bmin)
    
    else:
        None

# Loads hyperspectral data
path_pavia_data = "Pavia\PaviaU.mat"
path_pavia_gt = "Pavia\PaviaU_gt.mat"
pavia_data = scipy.io.loadmat(path_pavia_data)
pavia_gt = scipy.io.loadmat(path_pavia_gt)

print("Keys in PaviaU.mat:", pavia_data.keys())
print("Keys in PaviaU_gt.mat:", pavia_gt.keys())

hsi_data = pavia_data['paviaU']  # Hyperspectral image data
gt_labels = pavia_gt['paviaU_gt']  # Ground truth labels

print("HSI data shape:", hsi_data.shape)  
print("Ground truth shape:", gt_labels.shape)  

# Bands estimate
num_bands = 103
wavelengths = 430 + np.arange(num_bands) * 5

# Calculate statistics for each band
band_stats = []
for i, wavelength in enumerate(wavelengths):
    band_data = hsi_data[:, :, i].flatten()  # Flatten and remove masked values
    if band_data.size > 0:  # Check if there are valid pixels
        stats = {"Band": i + 1, "Wavelength (nm)": wavelength, "Mean": np.mean(band_data),
            "Std": np.std(band_data), "Min": np.min(band_data), "Max": np.max(band_data)}
        
        band_stats.append(stats)

# Print band statistics
for stats in band_stats:
    print(f"Band {stats['Band']} ({stats['Wavelength (nm)']} nm):")
    print(f"  Mean: {stats['Mean']:.2f}, Std: {stats['Std']:.2f}, Min: {stats['Min']:.2f}, Max: {stats['Max']:.2f}")
    print()

# Finds closest bands to visible spectrum
def find_closest_band(target_wavelength):
    return np.argmin(np.abs(np.array(wavelengths) - target_wavelength))

# RGB bands (using central wavelengths)
red_band = find_closest_band(650)  # ~650nm for red
green_band = find_closest_band(550)  # ~550nm for green
blue_band = find_closest_band(475)  # ~475nm for blue
  
# Extract bands
data_cube = pavia_data['paviaU']
red = data_cube[:, :, red_band].astype(float)
green = data_cube[:, :, green_band].astype(float)
blue = data_cube[:, :, blue_band].astype(float)

# Apply contrast stretching
red_stretched = stretch(red, stretch_type='percentile', percentiles=(5, 95))
green_stretched = stretch(green, stretch_type='percentile', percentiles=(5, 95))
blue_stretched = stretch(blue, stretch_type='percentile', percentiles=(5, 95))

# RGB image (stretch)
rgb_image = np.dstack((red_stretched, green_stretched, blue_stretched))
points = [(100, 50), (217, 243), (410, 190), (268.1, 163.3)]
material = ['grass', 'roofing', 'concrete', 'tarmac']

# Plot RGB image
plt.figure(figsize=(8, 6))
plt.imshow(rgb_image)
plt.title(f"Enhanced RGB Composite\n(Red Band {red_band}, Green Band {green_band}, Blue Band {blue_band})", fontsize=8)

for (x, y) in points:
    plt.scatter(y, x, color='red', s=50, marker='x', linewidths=1.5)
for i, (x, y) in enumerate(points):
    plt.text(y, x, material[i], color='cyan', fontsize=10, ha='right', va='bottom', 
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

plt.axis('on')
plt.savefig('pavia_RGB.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()  
for i, (x, y) in enumerate(points):
    ax = axes[i]
    x_int = int(round(x))  
    y_int = int(round(y))  
    spctrm = data_cube[x_int, y_int, :]  # Extract the spectrum at (x_int, y_int)
    spctrm_nrmlzd = (spctrm - np.min(spctrm)) / (np.max(spctrm) - np.min(spctrm))
    ax.plot(wavelengths[:-1], spctrm_nrmlzd[:-1], color='black')
    ax.set_title(f'{material[i]}')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    pass

# # Create a 3D figure
# rows, cols, bands = data_cube.shape
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')
# z_offset = 10  # gap between slices

# for b in range(bands):
#     band_data = data_cube[:, :, b] # Extract band b
#     X, Y = np.meshgrid(range(cols), range(rows)) # Create X, Y grid
#     Z = np.ones_like(band_data) * b * z_offset # Offset in z to stack slices
#     ax.plot_surface(X, Y, Z, rstride=5, cstride=5, facecolors=plt.cm.inferno(band_data / band_data.max()),
#                     linewidth=0, antialiased=False, shade=False)

# ax.set_xlabel('X (columns)')
# ax.set_ylabel('Y (rows)')
# ax.set_zlabel('Spectral band')
# ax.set_title('Hyperspectral Cube')
# ax.view_init(elev=30, azim=-60) 
# plt.tight_layout()
# plt.show()
