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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Bidirectional, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ECG BINARY CLASSIFICATION - FIXED DATA PREPROCESSING")
print("=" * 80)

# ============================================================================
# 1. LOAD & PREPROCESS DATASET (CORRECTED)
# ============================================================================
print("\n[1/6] Loading and preprocessing dataset...")
dataset_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/preprocessed_dataset/merged_ecg_dataset_all5_complete.npz"
data = np.load(dataset_path)
X = data['X']
y = data['y']

print(f"    Original X shape: {X.shape}")
print(f"    Original y unique values: {np.unique(y)}")
print(f"    Original y distribution: {np.bincount(y)}")

# Convert to binary: 0 = No arrhythmia (class 0), 1 = Any arrhythmia (classes 1-4)
y_binary = np.where(y > 0, 1, 0)
print(f"    Binary y distribution: {np.bincount(y_binary)}")

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
print(f"    Sampled y distribution: {np.bincount(y_sampled)}")

# *** FIX: Proper normalization for (N, 300) shape ***
# X_sampled is currently (500000, 300)
# Normalize across samples for each timestep (each column normalized independently)
X_flat_for_scaler = X_sampled.copy()  # Shape: (500000, 300)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_flat_for_scaler)  # Now StandardScaler works correctly

# Reshape for CNN: (N, timesteps, channels)
X_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
print(f"    Normalized X shape: {X_reshaped.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_sampled,
    test_size=0.2,
    stratify=y_sampled,
    random_state=42
)

print(f"    Train: {X_train.shape} | Test: {X_test.shape}")
print(f"    Train class dist: {np.bincount(y_train)}")
print(f"    Test class dist: {np.bincount(y_test)}")

# ============================================================================
# 2. ADVANCED 1D CNN MODEL
# ============================================================================
print("\n[2/6] Building 1D CNN model...")

model_cnn = Sequential([
    # Conv Block 1
    Conv1D(64, 3, activation='relu', padding='same', input_shape=(300, 1)),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Conv Block 2
    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Conv Block 3
    Conv1D(256, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Conv Block 4
    Conv1D(512, 3, activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    
    # Dense layers
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(f"    CNN Parameters: {model_cnn.count_params():,}")

# ============================================================================
# 3. TRAIN CNN
# ============================================================================
print("\n[3/6] Training 1D CNN...")

callbacks_cnn = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

history_cnn = model_cnn.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=callbacks_cnn,
    verbose=1
)

# ============================================================================
# 4. BUILD & TRAIN GRU MODEL
# ============================================================================
print("\n[4/6] Building Bidirectional GRU model...")

model_gru = Sequential([
    Input(shape=(300, 1)),
    Bidirectional(GRU(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    
    Bidirectional(GRU(32)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_gru.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

print(f"    GRU Parameters: {model_gru.count_params():,}")

print("\n[5/6] Training Bidirectional GRU...")

callbacks_gru = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

history_gru = model_gru.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    callbacks=callbacks_gru,
    verbose=1
)

# ============================================================================
# 6. ENSEMBLE PREDICTIONS & EVALUATION
# ============================================================================
print("\n[6/6] Ensemble voting and evaluation...")

# Get predictions from both models
y_pred_cnn = model_cnn.predict(X_test, verbose=0).flatten()
y_pred_gru = model_gru.predict(X_test, verbose=0).flatten()

# Ensemble: 60% CNN + 40% GRU
y_pred_ensemble = 0.6 * y_pred_cnn + 0.4 * y_pred_gru
y_pred_ensemble_binary = (y_pred_ensemble > 0.5).astype(int)

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_predictions(y_true, y_pred_proba, y_pred_binary, model_name):
    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n{'='*60}")
    print(f"{model_name} RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred_binary, target_names=['No Arrhythmia', 'Arrhythmia'])}")
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}

# Evaluate individual models
results_cnn = evaluate_predictions(y_test, y_pred_cnn, (y_pred_cnn > 0.5).astype(int), "1D CNN")
results_gru = evaluate_predictions(y_test, y_pred_gru, (y_pred_gru > 0.5).astype(int), "Bidirectional GRU")
results_ensemble = evaluate_predictions(y_test, y_pred_ensemble, y_pred_ensemble_binary, "ENSEMBLE (60% CNN + 40% GRU)")

# ============================================================================
# 7. SAVE MODELS & RESULTS
# ============================================================================
print("\n[SAVING] Models and results...")

# Create directories
os.makedirs("Binary_Classification/OneD_CNN", exist_ok=True)
os.makedirs("Binary_Classification/GRU", exist_ok=True)
os.makedirs("Binary_Classification/Ensemble", exist_ok=True)

# Save models
model_cnn.save("Binary_Classification/OneD_CNN/model_1d_cnn.h5")
model_gru.save("Binary_Classification/GRU/model_gru.h5")

# Save metrics
with open("Binary_Classification/OneD_CNN/metrics.json", "w") as f:
    json.dump(results_cnn, f, indent=2)
with open("Binary_Classification/GRU/metrics.json", "w") as f:
    json.dump(results_gru, f, indent=2)
with open("Binary_Classification/Ensemble/ensemble_metrics.json", "w") as f:
    json.dump(results_ensemble, f, indent=2)

# Save confusion matrices
def save_confusion_matrix(y_true, y_pred, filepath, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

save_confusion_matrix(y_test, (y_pred_cnn > 0.5).astype(int), "Binary_Classification/OneD_CNN/confusion_matrix.png", "1D CNN Confusion Matrix")
save_confusion_matrix(y_test, (y_pred_gru > 0.5).astype(int), "Binary_Classification/GRU/confusion_matrix.png", "GRU Confusion Matrix")
save_confusion_matrix(y_test, y_pred_ensemble_binary, "Binary_Classification/Ensemble/ensemble_confusion_matrix.png", "Ensemble Confusion Matrix")

# Save training history
def save_history_plot(history, filepath, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

save_history_plot(history_cnn, "Binary_Classification/OneD_CNN/training_history.png", "1D CNN")
save_history_plot(history_gru, "Binary_Classification/GRU/training_history.png", "GRU")

# Save predictions
np.savez("Binary_Classification/Ensemble/ensemble_predictions.npz", 
         y_test=y_test, 
         y_pred_ensemble=y_pred_ensemble,
         y_pred_cnn=y_pred_cnn,
         y_pred_gru=y_pred_gru)

print("\n✓ All models and results saved!")
print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best Ensemble Accuracy: {results_ensemble['accuracy']:.4f}")
print(f"Best Ensemble F1-Score: {results_ensemble['f1']:.4f}")
print(f"Best Ensemble ROC-AUC:  {results_ensemble['roc_auc']:.4f}")
