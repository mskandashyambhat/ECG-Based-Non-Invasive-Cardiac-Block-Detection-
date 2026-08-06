import numpy as np
from sklearn.preprocessing import StandardScaler

# Load preprocessed (normalized) data
dataset_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/preprocessed_dataset/merged_ecg_dataset_all5_complete.npz"
data = np.load(dataset_path)
X = data['X']
y = data['y']

print("Current preprocessed data:")
print(f"  Mean: {X.mean():.8f}, Std: {X.std():.8f}")
print(f"  Min: {X.min():.2f}, Max: {X.max():.2f}")

# Binary conversion
y_binary = np.where(y > 0, 1, 0)

# Check raw unprocessed data
raw_data_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/Unprocessed_Datasets/"
print(f"\nLooking for raw datasets in: {raw_data_path}")

import os
if os.path.exists(raw_data_path):
    datasets = os.listdir(raw_data_path)
    print(f"Available datasets: {datasets}")
else:
    print("Raw data path not found")

# The issue is: pre-normalized data with indistinguishable classes
# Solution: We need features that CAN distinguish the classes
# Options:
# 1. Use time-domain features (already tried - didn't work)
# 2. Use frequency-domain features (FFT)
# 3. Use wavelet transforms
# 4. Check if different input representation helps
# 5. Or, go back to raw data and do proper feature engineering

print("\n" + "=" * 80)
print("SOLUTION OPTIONS:")
print("=" * 80)
print("1. ✗ StandardScaler on normalized data - won't help (data already normalized)")
print("2. ✗ Use time-domain CNN - poor class separation")
print("3. ✓ Extract frequency-domain features (FFT)")
print("4. ✓ Use wavelet transform features")
print("5. ✓ Use multiple ECG signal characteristics as features")
print("\nRecommendation: Use frequency-domain analysis with CNN")
print("=" * 80)
