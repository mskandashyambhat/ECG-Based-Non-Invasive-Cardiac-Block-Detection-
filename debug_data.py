import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
dataset_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/preprocessed_dataset/merged_ecg_dataset_all5_complete.npz"
data = np.load(dataset_path)
X = data['X']
y = data['y']

print("=" * 80)
print("DATA INSPECTION")
print("=" * 80)

print(f"\nOriginal data shape: {X.shape}")
print(f"Original labels shape: {y.shape}")
print(f"Original label unique values: {np.unique(y)}")
print(f"Original label value counts: {np.bincount(y.astype(int))}")
print(f"First 20 labels: {y[:20]}")

# Binary conversion
y_binary = np.where(y > 0, 1, 0)
print(f"\nAfter binary conversion:")
print(f"Binary unique values: {np.unique(y_binary)}")
print(f"Binary value counts: {np.bincount(y_binary)}")
print(f"First 20 binary labels: {y_binary[:20]}")

# Sample 500k
sample_size = 500000
unique_labels = np.unique(y_binary)
samples_per_class = sample_size // len(unique_labels)

indices = np.array([], dtype=int)
for label in unique_labels:
    label_indices = np.where(y_binary == label)[0]
    sampled_indices = np.random.choice(label_indices, size=samples_per_class, replace=False)
    indices = np.concatenate([indices, sampled_indices])

np.random.shuffle(indices)
X_sampled = X[indices]
y_sampled = y_binary[indices]

print(f"\nAfter sampling 500k:")
print(f"Sampled X shape: {X_sampled.shape}")
print(f"Sampled y shape: {y_sampled.shape}")
print(f"Sampled y unique: {np.unique(y_sampled)}")
print(f"Sampled y counts: {np.bincount(y_sampled)}")

# Check X statistics
print(f"\nX statistics (first 100 samples):")
print(f"  Min: {X_sampled[:100].min():.6f}")
print(f"  Max: {X_sampled[:100].max():.6f}")
print(f"  Mean: {X_sampled[:100].mean():.6f}")
print(f"  Std: {X_sampled[:100].std():.6f}")

# Check class separation - are the two classes actually different?
class_0_samples = X_sampled[y_sampled == 0]
class_1_samples = X_sampled[y_sampled == 1]

print(f"\nClass 0 (No Block) statistics:")
print(f"  Min: {class_0_samples.min():.6f}")
print(f"  Max: {class_0_samples.max():.6f}")
print(f"  Mean: {class_0_samples.mean():.6f}")
print(f"  Std: {class_0_samples.std():.6f}")

print(f"\nClass 1 (Block Present) statistics:")
print(f"  Min: {class_1_samples.min():.6f}")
print(f"  Max: {class_1_samples.max():.6f}")
print(f"  Mean: {class_1_samples.mean():.6f}")
print(f"  Std: {class_1_samples.std():.6f}")

# Check if samples are actually different
mean_diff = np.abs(class_0_samples.mean() - class_1_samples.mean())
print(f"\nMean difference between classes: {mean_diff:.8f}")

if mean_diff < 0.001:
    print("⚠️  WARNING: Classes have almost identical means - they may not be separable!")
else:
    print("✓ Classes appear to be different")

# Normalization test
X_samples_2d = X_sampled.reshape(X_sampled.shape[0], -1)
print(f"\nNormalization test:")
print(f"Before scaling - Min: {X_samples_2d.min():.6f}, Max: {X_samples_2d.max():.6f}")

scaler = StandardScaler()
X_normalized_2d = scaler.fit_transform(X_samples_2d)
print(f"After scaling - Min: {X_normalized_2d.min():.6f}, Max: {X_normalized_2d.max():.6f}")
print(f"After scaling - Mean: {X_normalized_2d.mean():.6f}, Std: {X_normalized_2d.std():.6f}")

# Check for NaN or Inf
if np.any(np.isnan(X_normalized_2d)):
    print("⚠️  WARNING: NaN values found in normalized data!")
if np.any(np.isinf(X_normalized_2d)):
    print("⚠️  WARNING: Inf values found in normalized data!")
    
print("\n" + "=" * 80)
