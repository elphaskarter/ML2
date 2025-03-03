# --Data Interpretation--
import matplotlib.pyplot as plt
import spectral as spy
import numpy as np

def stretch(band, stretch_type='minmax', percentiles=(2, 98)):
    band_clean = band[~np.isnan(band)]
    if stretch_type == 'percentile':
        # Use percentile-based stretching
        p_low, p_high = np.nanpercentile(band_clean, percentiles)
        band = np.clip(band, p_low, p_high)
        return (band - p_low) / (p_high - p_low)
    
    elif stretch_type == 'minmax':
        # Classic min-max stretching
        bmin = np.nanmin(band_clean)
        bmax = np.nanmax(band_clean)
        return (band - bmin) / (bmax - bmin)
    
    else:
        raise ValueError("Invalid stretch_type. Use 'minmax' or 'percentile'")

# Loads hyperspectral data
hdr_file = r"C:\Users\elpha\OneDrive\Desktop\hsi_data\tait_hsi.hdr"
hsi_image = spy.open_image(hdr_file)
hsi_data = hsi_image.load()

# Extracts band wavelengths
bands = hsi_image.metadata.get("band names", [])
bands_float = [float(b.strip().replace(' ms', '').replace('nm', '').replace(',', '')) 
               for b in bands if b.strip()]

# Finds closest bands to visible spectrum
def find_closest_band(target_wavelength):
    return np.argmin(np.abs(np.array(bands_float) - target_wavelength))

# Get RGB bands (using central wavelengths)
red_band = find_closest_band(650)  # ~650nm for red
green_band = find_closest_band(550)  # ~550nm for green
blue_band = find_closest_band(475)  # ~475nm for blue
  
# Extract bands with stretching
red = hsi_data[:, :, red_band].astype(float)
green = hsi_data[:, :, green_band].astype(float)
blue = hsi_data[:, :, blue_band].astype(float)

# Apply contrast stretching (try different parameters)
red_stretched = stretch(red, stretch_type='percentile', percentiles=(5, 95))
green_stretched = stretch(green, stretch_type='percentile', percentiles=(5, 95))
blue_stretched = stretch(blue, stretch_type='percentile', percentiles=(5, 95))

# Combine into RGB image
rgb_image = np.dstack((red_stretched, green_stretched, blue_stretched))

# Plot and save
plt.figure(figsize=(8, 6))
plt.imshow(rgb_image)
plt.title(f"Enhanced RGB Composite\n(Red Band {red_band}, Green Band {green_band}, Blue Band {blue_band})")
plt.axis('off')
plt.savefig('Sentinel_2_RGB.png', dpi=300, bbox_inches='tight')
plt.show()

# hyperspectral data to resize (250 by 250)
rows, cols, n_bands = hsi_data.shape
print(f"HSI data shape: {rows} x {cols} x {n_bands}")

row_start = 10
col_start = 300
patch_size = 250

patch = hsi_data[row_start:row_start + patch_size, col_start:col_start + patch_size, :]

# Get RGB bands (using central wavelengths)
# Extract bands with stretching
red_patch = patch[:, :, red_band].astype(float)
green_patch = patch[:, :, green_band].astype(float)
blue_patch = patch[:, :, blue_band].astype(float)

# Apply contrast stretching (try different parameters)
red_stretched_patch = stretch(red_patch, stretch_type='percentile', percentiles=(5, 95))
green_stretched_patch = stretch(green_patch, stretch_type='percentile', percentiles=(5, 95))
blue_stretched_patch = stretch(blue_patch, stretch_type='percentile', percentiles=(5, 95))

# Combine into RGB image
rgb_image_patch = np.dstack((red_stretched_patch, green_stretched_patch, blue_stretched_patch))

# Plot and save
plt.figure(figsize=(8, 6))
plt.imshow(rgb_image_patch)
plt.title(f"RGB Composite Patch\n(Red Band {red_band}, Green Band {green_band}, Blue Band {blue_band})")
plt.axis('off')
plt.savefig('Patch_RGB_Sentinel_2.png', dpi=300, bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    pass

# END