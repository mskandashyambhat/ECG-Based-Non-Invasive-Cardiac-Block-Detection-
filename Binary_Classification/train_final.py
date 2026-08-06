import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ECG BINARY CLASSIFICATION - DIRECT TRAINING (NO EXTRA PREPROCESSING)")
print("=" * 80)

# ============================================================================
# 1. LOAD PREPROCESSED DATASET (NO EXTRA PREPROCESSING)
# ============================================================================
print("\n[1/6] Loading preprocessed dataset (no extra preprocessing)...")
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

# ✅ NOTE: Data is already pre-normalized in the dataset!
# Don't apply StandardScaler - it's already normalized
# Just reshape for CNN/RNN
X_reshaped = X_sampled.reshape(X_sampled.shape[0], X_sampled.shape[1], 1)
print(f"    Final shape for model: {X_reshaped.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_sampled,
    test_size=0.2,
    stratify=y_sampled,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

print(f"    Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"    Train class dist: {np.bincount(y_train)}")
print(f"    Val class dist: {np.bincount(y_val)}")


def tune_threshold(y_true, y_proba):
    """Pick the cutoff that best balances accuracy and binary F1 on validation data."""
    best_threshold = 0.5
    best_score = -1.0
    best_accuracy = 0.0
    best_f1 = 0.0

    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_proba >= threshold).astype(int).flatten()
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        score = 0.5 * accuracy + 0.5 * f1

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_accuracy = float(accuracy)
            best_f1 = float(f1)

    return best_threshold, best_accuracy, best_f1

# ============================================================================
# 2. ADVANCED 1D CNN MODEL (HIGH CAPACITY)
# ============================================================================
print("\n[2/5] Building 1D CNN model...")

model_cnn = Sequential([
    Input(shape=(300, 1)),
    Conv1D(32, 7, activation='relu', padding='same'),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(64, 5, activation='relu', padding='same'),
    MaxPooling1D(2),
    Dropout(0.2),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),

    GlobalAveragePooling1D(),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(f"    CNN Parameters: {model_cnn.count_params():,}")

callbacks_cnn = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

print("    Training 1D CNN...")
history_cnn = model_cnn.fit(
    X_train, y_train,
    batch_size=256,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=callbacks_cnn,
    verbose=1
)

# ============================================================================
# 3. EVALUATE 1D CNN
# ============================================================================
print("\n[3/5] Evaluating 1D CNN...")
y_val_cnn_proba = model_cnn.predict(X_val, verbose=0)
cnn_threshold, cnn_val_acc, cnn_val_f1 = tune_threshold(y_val, y_val_cnn_proba)
print(f"    Best validation threshold for CNN: {cnn_threshold:.3f} | val_acc={cnn_val_acc:.4f} | val_f1={cnn_val_f1:.4f}")

y_pred_cnn_proba = model_cnn.predict(X_test, verbose=0)
y_pred_cnn = (y_pred_cnn_proba >= cnn_threshold).astype(int).flatten()

test_loss_cnn, _, _ = model_cnn.evaluate(X_test, y_test, verbose=0)
test_acc_cnn = accuracy_score(y_test, y_pred_cnn)
roc_auc_cnn = roc_auc_score(y_test, y_pred_cnn_proba)

prec_cnn, rec_cnn, f1_cnn, _ = precision_recall_fscore_support(y_test, y_pred_cnn, average='binary', zero_division=0)

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
print("\n[4/5] Building GRU model...")

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
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(f"    GRU Parameters: {model_gru.count_params():,}")

callbacks_gru = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

print("    Training GRU...")
history_gru = model_gru.fit(
    X_train, y_train,
    batch_size=512,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=callbacks_gru,
    verbose=1
)

# ============================================================================
# 5. EVALUATE GRU
# ============================================================================
print("\n[5/5] Evaluating GRU...")
y_val_gru_proba = model_gru.predict(X_val, verbose=0)
gru_threshold, gru_val_acc, gru_val_f1 = tune_threshold(y_val, y_val_gru_proba)
print(f"    Best validation threshold for GRU: {gru_threshold:.3f} | val_acc={gru_val_acc:.4f} | val_f1={gru_val_f1:.4f}")

y_pred_gru_proba = model_gru.predict(X_test, verbose=0)
y_pred_gru = (y_pred_gru_proba >= gru_threshold).astype(int).flatten()

test_loss_gru, _, _ = model_gru.evaluate(X_test, y_test, verbose=0)
test_acc_gru = accuracy_score(y_test, y_pred_gru)
roc_auc_gru = roc_auc_score(y_test, y_pred_gru_proba)

prec_gru, rec_gru, f1_gru, _ = precision_recall_fscore_support(y_test, y_pred_gru, average='binary', zero_division=0)

print(f"\n    📊 GRU RESULTS:")
print(f"    - Accuracy: {test_acc_gru:.4f}")
print(f"    - Precision: {prec_gru:.4f}")
print(f"    - Recall: {rec_gru:.4f}")
print(f"    - F1-Score: {f1_gru:.4f}")
print(f"    - ROC-AUC: {roc_auc_gru:.4f}")
print(f"    - Loss: {test_loss_gru:.4f}")

# ============================================================================
# SAVE MODELS & RESULTS
# ============================================================================
base_dir = "/Users/skandashyam/Desktop/MajorProject/Project/Binary_Classification"

# ✅ Save 1D CNN Model
print("\n    Saving 1D CNN model...")
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
    'best_threshold': float(cnn_threshold),
    'validation_accuracy_at_best_threshold': float(cnn_val_acc),
    'validation_f1_at_best_threshold': float(cnn_val_f1),
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
axes[0].set_title('1D CNN - Accuracy')
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
print(f"    ✅ CNN model saved to {cnn_dir}")

# ✅ Save GRU Model
print("    Saving GRU model...")
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
    'best_threshold': float(gru_threshold),
    'validation_accuracy_at_best_threshold': float(gru_val_acc),
    'validation_f1_at_best_threshold': float(gru_val_f1),
    'confusion_matrix': cm_gru.tolist(),
    'parameters': model_gru.count_params()
}
with open(os.path.join(gru_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_gru, f, indent=4)

# Plot GRU
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history_gru.history['accuracy'], label='Train Acc', linewidth=2)
axes[0].plot(history_gru.history['val_accuracy'], label='Val Acc', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Bidirectional GRU - Accuracy')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history_gru.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history_gru.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Bidirectional GRU - Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(gru_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✅ GRU model saved to {gru_dir}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - FINAL RESULTS SUMMARY")
print("=" * 80)

summary_data = {
    'timestamp': datetime.now().isoformat(),
    'models': {
        '1D_CNN': {
            'accuracy': float(test_acc_cnn),
            'precision': float(prec_cnn),
            'recall': float(rec_cnn),
            'f1_score': float(f1_cnn),
            'roc_auc': float(roc_auc_cnn),
            'best_threshold': float(cnn_threshold),
            'loss': float(test_loss_cnn),
            'parameters': model_cnn.count_params()
        },
        'GRU': {
            'accuracy': float(test_acc_gru),
            'precision': float(prec_gru),
            'recall': float(rec_gru),
            'f1_score': float(f1_gru),
            'roc_auc': float(roc_auc_gru),
            'best_threshold': float(gru_threshold),
            'loss': float(test_loss_gru),
            'parameters': model_gru.count_params()
        }
    }
}

summary_file = os.path.join(base_dir, 'training_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary_data, f, indent=4)

print(f"\n📊 1D CNN:        Accuracy={test_acc_cnn:.4f} | F1={f1_cnn:.4f} | ROC-AUC={roc_auc_cnn:.4f}")
print(f"📊 GRU:           Accuracy={test_acc_gru:.4f} | F1={f1_gru:.4f} | ROC-AUC={roc_auc_gru:.4f}")
print(f"\n✅ Models saved:")
print(f"   - 1D CNN: {cnn_dir}/model_1d_cnn.h5")
print(f"   - GRU: {gru_dir}/model_gru.h5")
print(f"\n✅ Summary: {summary_file}")
print("=" * 80)
