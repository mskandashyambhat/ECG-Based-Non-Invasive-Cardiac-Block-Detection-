import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, GRU, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ECG BINARY CLASSIFICATION - HIGH ACCURACY OPTIMIZATION")
print("=" * 80)

# ============================================================================
# 1. LOAD & PREPROCESS DATASET (LARGER + NORMALIZED)
# ============================================================================
print("\n[1/6] Loading and preprocessing dataset...")
dataset_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/preprocessed_dataset/merged_ecg_dataset_all5_complete.npz"
data = np.load(dataset_path)
X = data['X']
y = data['y']

print(f"    Original X shape: {X.shape}")

# Convert to binary
y_binary = np.where(y > 0, 1, 0)

# Use 500k samples for better performance
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

print(f"    Sampled X shape: {X_sampled.shape}")
print(f"    Class distribution: {np.bincount(y_sampled)}")

# Normalize (Z-score standardization)
scaler = StandardScaler()
X_flat = X_sampled.reshape(-1, X_sampled.shape[1])
X_normalized = scaler.fit_transform(X_flat)
X_normalized = X_normalized.reshape(X_sampled.shape)

# Reshape for CNN/RNN
X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_sampled,
    test_size=0.2,
    stratify=y_sampled,
    random_state=42
)

print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
print(f"    Train class dist: {np.bincount(y_train)}")

# ============================================================================
# 2. ADVANCED 1D CNN MODEL (HIGH CAPACITY)
# ============================================================================
print("\n[2/6] Building advanced 1D CNN model...")

model_cnn = Sequential([
    Input(shape=(300, 1)),
    
    Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    MaxPooling1D(2),
    Dropout(0.25),
    
    Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    MaxPooling1D(2),
    Dropout(0.25),
    
    Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    MaxPooling1D(2),
    Dropout(0.25),
    
    Conv1D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    
    Dense(256, activation='relu', kernel_regularizer=l2(1e-5)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-5)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(f"    CNN Parameters: {model_cnn.count_params():,}")

callbacks_cnn = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

print("    Training 1D CNN...")
history_cnn = model_cnn.fit(
    X_train, y_train,
    batch_size=128,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks_cnn,
    verbose=1
)

# ============================================================================
# 3. EVALUATE 1D CNN
# ============================================================================
print("\n[3/6] Evaluating 1D CNN...")
y_pred_cnn_proba = model_cnn.predict(X_test, verbose=0)
y_pred_cnn = (y_pred_cnn_proba > 0.5).astype(int).flatten()

test_loss_cnn, test_acc_cnn, test_auc_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)
roc_auc_cnn = roc_auc_score(y_test, y_pred_cnn_proba)

prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(y_test, y_pred_cnn, average='weighted')

print(f"\n    📊 1D CNN RESULTS:")
print(f"    - Accuracy: {test_acc_cnn:.4f}")
print(f"    - Precision: {prec_cnn:.4f}")
print(f"    - Recall: {rec_cnn:.4f}")
print(f"    - F1-Score: {f1_cnn:.4f}")
print(f"    - ROC-AUC: {roc_auc_cnn:.4f}")
print(f"    - Loss: {test_loss_cnn:.4f}")

# ============================================================================
# 4. ADVANCED GRU MODEL (OPTIMIZED)
# ============================================================================
print("\n[4/6] Building advanced GRU model...")

model_gru = Sequential([
    Input(shape=(300, 1)),
    
    Bidirectional(GRU(64, return_sequences=True, dropout=0.2)),
    BatchNormalization(),
    Dropout(0.2),
    
    Bidirectional(GRU(128, return_sequences=True, dropout=0.2)),
    BatchNormalization(),
    Dropout(0.2),
    
    Bidirectional(GRU(64, return_sequences=False, dropout=0.2)),
    Dropout(0.3),
    
    Dense(256, activation='relu', kernel_regularizer=l2(1e-5)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-5)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_gru.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(f"    GRU Parameters: {model_gru.count_params():,}")

callbacks_gru = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
]

print("    Training GRU...")
history_gru = model_gru.fit(
    X_train, y_train,
    batch_size=128,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks_gru,
    verbose=1
)

# ============================================================================
# 5. EVALUATE GRU
# ============================================================================
print("\n[5/6] Evaluating GRU...")
y_pred_gru_proba = model_gru.predict(X_test, verbose=0)
y_pred_gru = (y_pred_gru_proba > 0.5).astype(int).flatten()

test_loss_gru, test_acc_gru, test_auc_gru = model_gru.evaluate(X_test, y_test, verbose=0)
roc_auc_gru = roc_auc_score(y_test, y_pred_gru_proba)

prec_gru, rec_gru, f1_gru, _ = precision_recall_fscore_support(y_test, y_pred_gru, average='weighted')

print(f"\n    📊 GRU RESULTS:")
print(f"    - Accuracy: {test_acc_gru:.4f}")
print(f"    - Precision: {prec_gru:.4f}")
print(f"    - Recall: {rec_gru:.4f}")
print(f"    - F1-Score: {f1_gru:.4f}")
print(f"    - ROC-AUC: {roc_auc_gru:.4f}")
print(f"    - Loss: {test_loss_gru:.4f}")

# ============================================================================
# 6. ENSEMBLE & SAVE RESULTS
# ============================================================================
print("\n[6/6] Creating ensemble predictions...")

# Ensemble: Average predictions
y_pred_ensemble_proba = 0.6 * y_pred_cnn_proba + 0.4 * y_pred_gru_proba
y_pred_ensemble = (y_pred_ensemble_proba > 0.5).astype(int).flatten()

roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble_proba)
prec_ens, rec_ens, f1_ens, _ = precision_recall_fscore_support(y_test, y_pred_ensemble, average='weighted')
acc_ens = np.mean(y_pred_ensemble == y_test)

print(f"\n    🎯 ENSEMBLE RESULTS:")
print(f"    - Accuracy: {acc_ens:.4f}")
print(f"    - Precision: {prec_ens:.4f}")
print(f"    - Recall: {rec_ens:.4f}")
print(f"    - F1-Score: {f1_ens:.4f}")
print(f"    - ROC-AUC: {roc_auc_ensemble:.4f}")

# Save models
base_dir = "/Users/skandashyam/Desktop/MajorProject/Project/Binary_Classification"

# Save 1D CNN
cnn_dir = os.path.join(base_dir, "OneD_CNN")
os.makedirs(cnn_dir, exist_ok=True)

model_cnn.save(os.path.join(cnn_dir, "model_1d_cnn.h5"))
with open(os.path.join(cnn_dir, "training_history.pkl"), 'wb') as f:
    pickle.dump(history_cnn.history, f)
np.savez(os.path.join(cnn_dir, "predictions.npz"),
         y_true=y_test, y_pred=y_pred_cnn, y_pred_proba=y_pred_cnn_proba)

cm_cnn = confusion_matrix(y_test, y_pred_cnn)
metrics_cnn = {
    'model': '1D CNN (Advanced)',
    'test_accuracy': float(test_acc_cnn),
    'test_loss': float(test_loss_cnn),
    'precision': float(prec_cnn),
    'recall': float(rec_cnn),
    'f1_score': float(f1_cnn),
    'roc_auc': float(roc_auc_cnn),
    'confusion_matrix': cm_cnn.tolist(),
    'parameters': model_cnn.count_params()
}
with open(os.path.join(cnn_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_cnn, f, indent=4)

# Plot CNN
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_cnn.history['accuracy'], label='Train Acc', linewidth=2)
axes[0].plot(history_cnn.history['val_accuracy'], label='Val Acc', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('1D CNN - Accuracy (Target: 97-98%)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history_cnn.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history_cnn.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('1D CNN - Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cnn_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Block', 'Block Present'],
            yticklabels=['No Block', 'Block Present'],
            ax=ax, cbar=False)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('1D CNN - Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(cnn_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ 1D CNN model saved to {cnn_dir}/")

# Save GRU
gru_dir = os.path.join(base_dir, "GRU")
os.makedirs(gru_dir, exist_ok=True)

model_gru.save(os.path.join(gru_dir, "model_gru.h5"))
with open(os.path.join(gru_dir, "training_history.pkl"), 'wb') as f:
    pickle.dump(history_gru.history, f)
np.savez(os.path.join(gru_dir, "predictions.npz"),
         y_true=y_test, y_pred=y_pred_gru, y_pred_proba=y_pred_gru_proba)

cm_gru = confusion_matrix(y_test, y_pred_gru)
metrics_gru = {
    'model': 'Bidirectional GRU (Advanced)',
    'test_accuracy': float(test_acc_gru),
    'test_loss': float(test_loss_gru),
    'precision': float(prec_gru),
    'recall': float(rec_gru),
    'f1_score': float(f1_gru),
    'roc_auc': float(roc_auc_gru),
    'confusion_matrix': cm_gru.tolist(),
    'parameters': model_gru.count_params()
}
with open(os.path.join(gru_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_gru, f, indent=4)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_gru.history['accuracy'], label='Train Acc', linewidth=2)
axes[0].plot(history_gru.history['val_accuracy'], label='Val Acc', linewidth=2)
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

print(f"✓ GRU model saved to {gru_dir}/")

# Save Ensemble
ensemble_dir = os.path.join(base_dir, "Ensemble")
os.makedirs(ensemble_dir, exist_ok=True)

np.savez(os.path.join(ensemble_dir, "predictions.npz"),
         y_true=y_test, y_pred=y_pred_ensemble, y_pred_proba=y_pred_ensemble_proba)

cm_ens = confusion_matrix(y_test, y_pred_ensemble)
metrics_ens = {
    'model': 'CNN + GRU Ensemble (60-40 weighted)',
    'test_accuracy': float(acc_ens),
    'precision': float(prec_ens),
    'recall': float(rec_ens),
    'f1_score': float(f1_ens),
    'roc_auc': float(roc_auc_ensemble),
    'confusion_matrix': cm_ens.tolist()
}
with open(os.path.join(ensemble_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_ens, f, indent=4)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Block', 'Block Present'],
            yticklabels=['No Block', 'Block Present'],
            ax=ax, cbar=False)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('Ensemble - Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(ensemble_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Ensemble results saved to {ensemble_dir}/")

# Final summary
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 FINAL RESULTS:")
print(f"\n1D CNN:")
print(f"  Accuracy: {test_acc_cnn:.4f} | Precision: {prec_cnn:.4f} | Recall: {rec_cnn:.4f} | F1: {f1_cnn:.4f}")
print(f"\nBidirectional GRU:")
print(f"  Accuracy: {test_acc_gru:.4f} | Precision: {prec_gru:.4f} | Recall: {rec_gru:.4f} | F1: {f1_gru:.4f}")
print(f"\n🎯 ENSEMBLE (Best):")
print(f"  Accuracy: {acc_ens:.4f} | Precision: {prec_ens:.4f} | Recall: {rec_ens:.4f} | F1: {f1_ens:.4f}")
print(f"\n✓ All models saved to: {base_dir}/")
