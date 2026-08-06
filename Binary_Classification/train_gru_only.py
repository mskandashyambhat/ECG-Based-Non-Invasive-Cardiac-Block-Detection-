import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ECG BINARY CLASSIFICATION - GRU MODEL ONLY (OPTIMIZED)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1/4] Loading dataset...")
dataset_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/preprocessed_dataset/merged_ecg_dataset_all5_complete.npz"
data = np.load(dataset_path)
X = data['X']
y = data['y']

print(f"    Original X shape: {X.shape}")
print(f"    Original y shape: {y.shape}")

# ============================================================================
# 2. CONVERT LABELS TO BINARY & SAMPLE
# ============================================================================
print("\n[2/4] Converting to binary and sampling...")
y_binary = np.where(y > 0, 1, 0)

# Stratified sampling for 300k samples (fast training)
sample_size = 300000
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

# Reshape
X_reshaped = X_sampled.reshape(X_sampled.shape[0], X_sampled.shape[1], 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_sampled, 
    test_size=0.2, 
    stratify=y_sampled,
    random_state=42
)

print(f"    X_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"    y_train distribution: {np.bincount(y_train)}")
print(f"    y_test distribution: {np.bincount(y_test)}")

# ============================================================================
# 3. BUILD OPTIMIZED GRU MODEL
# ============================================================================
print("\n[3/4] Building optimized GRU model...")

model_gru = Sequential([
    Input(shape=(300, 1)),
    
    GRU(32, return_sequences=True, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    
    GRU(64, return_sequences=False, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_gru.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"    Model parameters: {model_gru.count_params():,}")

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

# ============================================================================
# 4. TRAIN GRU
# ============================================================================
print("\n[4/4] Training GRU model...")
print("-" * 80)

history_gru = model_gru.fit(
    X_train, y_train,
    batch_size=512,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================================================
# 5. EVALUATE
# ============================================================================
print("\n" + "=" * 80)
print("GRU MODEL EVALUATION")
print("=" * 80)

y_pred_gru = (model_gru.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
y_pred_proba_gru = model_gru.predict(X_test, verbose=0).flatten()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gru, 
                          target_names=['No Block', 'Block Present']))

cm_gru = confusion_matrix(y_test, y_pred_gru)
print("\nConfusion Matrix:")
print(cm_gru)

roc_auc_gru = roc_auc_score(y_test, y_pred_proba_gru)
print(f"\nROC-AUC Score: {roc_auc_gru:.4f}")

test_loss_gru, test_acc_gru = model_gru.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc_gru:.4f}")
print(f"Test Loss: {test_loss_gru:.4f}")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

base_dir = "/Users/skandashyam/Desktop/MajorProject/Project/Binary_Classification"
gru_dir = os.path.join(base_dir, "GRU")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save model
model_gru.save(os.path.join(gru_dir, "model_gru.h5"))
print(f"✓ GRU model saved")

# Save history
with open(os.path.join(gru_dir, "training_history.pkl"), 'wb') as f:
    pickle.dump(history_gru.history, f)
print(f"✓ Training history saved")

# Save metrics
metrics_gru = {
    'model': 'GRU (Optimized)',
    'test_accuracy': float(test_acc_gru),
    'test_loss': float(test_loss_gru),
    'roc_auc': float(roc_auc_gru),
    'confusion_matrix': cm_gru.tolist(),
    'classification_report': classification_report(y_test, y_pred_gru, 
                                                   target_names=['No Block', 'Block Present'],
                                                   output_dict=True)
}

with open(os.path.join(gru_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_gru, f, indent=4)
print(f"✓ Metrics saved")

# Save predictions
np.savez(os.path.join(gru_dir, "predictions.npz"),
         y_true=y_test,
         y_pred=y_pred_gru,
         y_pred_proba=y_pred_proba_gru)
print(f"✓ Predictions saved")

# Save plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_gru.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history_gru.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('GRU - Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history_gru.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history_gru.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('GRU - Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(gru_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_gru, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Block', 'Block Present'],
            yticklabels=['No Block', 'Block Present'],
            ax=ax, cbar=False)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('GRU - Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(gru_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Plots saved")

print("\n" + "=" * 80)
print("✓ GRU TRAINING COMPLETE!")
print("=" * 80)
print(f"\nResults saved in: {gru_dir}")
print(f"Test Accuracy: {test_acc_gru:.4f}")
print(f"ROC-AUC: {roc_auc_gru:.4f}")
