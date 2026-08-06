import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, GRU, Dense, BatchNormalization, MaxPooling1D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("ECG BINARY CLASSIFICATION - TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1/6] Loading dataset...")
dataset_path = "/Users/skandashyam/Desktop/MajorProject/Project/Dataset/preprocessed_dataset/merged_ecg_dataset_all5_complete.npz"
data = np.load(dataset_path)
X = data['X']
y = data['y']

print(f"    Original X shape: {X.shape}")
print(f"    Original y shape: {y.shape}")
print(f"    Original label distribution: {np.bincount(y)}")

# ============================================================================
# 2. CONVERT LABELS TO BINARY
# ============================================================================
print("\n[2/6] Converting labels to binary...")
# 0 = Normal (No Block)
# 1-4 = Block Present (AV Block, Complete Heart Block, RBBB, LBBB)
y_binary = np.where(y > 0, 1, 0)

print(f"    Binary label distribution:")
print(f"    - Class 0 (No Block): {np.sum(y_binary == 0)}")
print(f"    - Class 1 (Block Present): {np.sum(y_binary == 1)}")

# ============================================================================
# 3. SAMPLE DATASET FOR EFFICIENT TRAINING
# ============================================================================
print("\n[3/6] Sampling dataset for efficient training...")
# Using stratified sampling for 300k samples for faster training
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

print(f"    Sampled X shape: {X_sampled.shape}")
print(f"    Sampled y distribution: {np.bincount(y_sampled)}")

# ============================================================================
# 4. RESHAPE INPUT FOR DEEP LEARNING
# ============================================================================
print("\n[4/6] Reshaping input data...")
X_reshaped = X_sampled.reshape(X_sampled.shape[0], X_sampled.shape[1], 1)
print(f"    Reshaped X: {X_reshaped.shape}")

# ============================================================================
# 5. STRATIFIED TRAIN-TEST SPLIT (80-20)
# ============================================================================
print("\n[5/6] Splitting dataset (80-20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_sampled, 
    test_size=0.2, 
    stratify=y_sampled,
    random_state=42
)

print(f"    X_train shape: {X_train.shape}")
print(f"    X_test shape: {X_test.shape}")
print(f"    y_train distribution: {np.bincount(y_train)}")
print(f"    y_test distribution: {np.bincount(y_test)}")

# ============================================================================
# 6. BUILD MODELS
# ============================================================================

# CALLBACKS
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

# MODEL 1: 1D CNN
print("\n[6a/6] Building 1D CNN Model...")

model_1d_cnn = Sequential([
    Input(shape=(300, 1)),
    
    # Block 1
    Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    # Block 2
    Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    
    # Block 3
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # Block 4
    Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # Global pooling and dense layers
    tf.keras.layers.GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_1d_cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"    Model parameters: {model_1d_cnn.count_params():,}")

# MODEL 2: GRU (Optimized for speed)
print("\n[6b/6] Building GRU Model...")

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

# ============================================================================
# 7. TRAIN MODELS
# ============================================================================
print("\n[7/7] Training models...")

# Train 1D CNN
print("\n" + "-" * 80)
print("Training 1D CNN Model...")
print("-" * 80)

history_1d_cnn = model_1d_cnn.fit(
    X_train, y_train,
    batch_size=256,
    epochs=30,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Train GRU
print("\n" + "-" * 80)
print("Training GRU Model...")
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
# 7. EVALUATE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)

# Predictions
y_pred_1d_cnn = (model_1d_cnn.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
y_pred_proba_1d_cnn = model_1d_cnn.predict(X_test, verbose=0).flatten()

y_pred_gru = (model_gru.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
y_pred_proba_gru = model_gru.predict(X_test, verbose=0).flatten()

# Evaluate 1D CNN
print("\n" + "-" * 80)
print("1D CNN MODEL EVALUATION")
print("-" * 80)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_1d_cnn, 
                          target_names=['No Block', 'Block Present']))

cm_1d_cnn = confusion_matrix(y_test, y_pred_1d_cnn)
print("\nConfusion Matrix:")
print(cm_1d_cnn)

roc_auc_1d_cnn = roc_auc_score(y_test, y_pred_proba_1d_cnn)
print(f"\nROC-AUC Score: {roc_auc_1d_cnn:.4f}")

# Test accuracy
test_loss_1d_cnn, test_acc_1d_cnn = model_1d_cnn.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc_1d_cnn:.4f}")
print(f"Test Loss: {test_loss_1d_cnn:.4f}")

# Evaluate GRU
print("\n" + "-" * 80)
print("GRU MODEL EVALUATION")
print("-" * 80)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gru, 
                          target_names=['No Block', 'Block Present']))

cm_gru = confusion_matrix(y_test, y_pred_gru)
print("\nConfusion Matrix:")
print(cm_gru)

roc_auc_gru = roc_auc_score(y_test, y_pred_proba_gru)
print(f"\nROC-AUC Score: {roc_auc_gru:.4f}")

# Test accuracy
test_loss_gru, test_acc_gru = model_gru.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc_gru:.4f}")
print(f"Test Loss: {test_loss_gru:.4f}")

# ============================================================================
# 8. SAVE MODELS AND RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODELS AND RESULTS")
print("=" * 80)

base_dir = "/Users/skandashyam/Desktop/MajorProject/Project/Binary_Classification"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save 1D CNN
cnn_dir = os.path.join(base_dir, "OneD_CNN")
model_1d_cnn.save(os.path.join(cnn_dir, "model_1d_cnn.h5"))
print(f"\n✓ 1D CNN model saved to {cnn_dir}/model_1d_cnn.h5")

# Save GRU
gru_dir = os.path.join(base_dir, "GRU")
model_gru.save(os.path.join(gru_dir, "model_gru.h5"))
print(f"✓ GRU model saved to {gru_dir}/model_gru.h5")

# Save histories
with open(os.path.join(cnn_dir, "training_history.pkl"), 'wb') as f:
    pickle.dump(history_1d_cnn.history, f)
print(f"✓ 1D CNN training history saved")

with open(os.path.join(gru_dir, "training_history.pkl"), 'wb') as f:
    pickle.dump(history_gru.history, f)
print(f"✓ GRU training history saved")

# Save predictions and metrics
metrics_1d_cnn = {
    'model': '1D CNN',
    'test_accuracy': float(test_acc_1d_cnn),
    'test_loss': float(test_loss_1d_cnn),
    'roc_auc': float(roc_auc_1d_cnn),
    'confusion_matrix': cm_1d_cnn.tolist(),
    'classification_report': classification_report(y_test, y_pred_1d_cnn, 
                                                   target_names=['No Block', 'Block Present'],
                                                   output_dict=True)
}

metrics_gru = {
    'model': 'GRU',
    'test_accuracy': float(test_acc_gru),
    'test_loss': float(test_loss_gru),
    'roc_auc': float(roc_auc_gru),
    'confusion_matrix': cm_gru.tolist(),
    'classification_report': classification_report(y_test, y_pred_gru, 
                                                   target_names=['No Block', 'Block Present'],
                                                   output_dict=True)
}

with open(os.path.join(cnn_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_1d_cnn, f, indent=4)
print(f"✓ 1D CNN metrics saved")

with open(os.path.join(gru_dir, "metrics.json"), 'w') as f:
    json.dump(metrics_gru, f, indent=4)
print(f"✓ GRU metrics saved")

# Save predictions
np.savez(os.path.join(cnn_dir, "predictions.npz"),
         y_true=y_test,
         y_pred=y_pred_1d_cnn,
         y_pred_proba=y_pred_proba_1d_cnn)
print(f"✓ 1D CNN predictions saved")

np.savez(os.path.join(gru_dir, "predictions.npz"),
         y_true=y_test,
         y_pred=y_pred_gru,
         y_pred_proba=y_pred_proba_gru)
print(f"✓ GRU predictions saved")

# Generate and save plots
def save_plots(history, predictions, y_true, cm, model_name, save_dir):
    """Generate and save training plots and confusion matrix"""
    
    # Training history plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Block', 'Block Present'],
                yticklabels=['No Block', 'Block Present'],
                ax=ax, cbar=False)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

save_plots(history_1d_cnn.history, y_pred_proba_1d_cnn, y_test, cm_1d_cnn, '1D CNN', cnn_dir)
save_plots(history_gru.history, y_pred_proba_gru, y_test, cm_gru, 'GRU', gru_dir)

print(f"✓ 1D CNN plots saved")
print(f"✓ GRU plots saved")

# Save summary report
summary = f"""
ECG BINARY CLASSIFICATION - TRAINING SUMMARY
{'=' * 80}
Timestamp: {timestamp}

DATASET INFORMATION:
- Total samples: {len(X_reshaped):,}
- Training samples: {len(X_train):,}
- Test samples: {len(X_test):,}
- Input shape: (300, 1)
- Classes: 2 (No Block, Block Present)

TRAINING CONFIGURATION:
- Batch size: 256
- Epochs: 30
- Validation split: 0.2
- Optimizer: Adam (learning_rate=0.001)
- Loss function: binary_crossentropy
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3)

{'=' * 80}
1D CNN MODEL RESULTS:
{'=' * 80}
Test Accuracy: {test_acc_1d_cnn:.4f}
Test Loss: {test_loss_1d_cnn:.4f}
ROC-AUC Score: {roc_auc_1d_cnn:.4f}
Total Parameters: {model_1d_cnn.count_params():,}

Confusion Matrix:
{cm_1d_cnn}

Classification Report:
{classification_report(y_test, y_pred_1d_cnn, target_names=['No Block', 'Block Present'])}

{'=' * 80}
GRU MODEL RESULTS:
{'=' * 80}
Test Accuracy: {test_acc_gru:.4f}
Test Loss: {test_loss_gru:.4f}
ROC-AUC Score: {roc_auc_gru:.4f}
Total Parameters: {model_gru.count_params():,}

Confusion Matrix:
{cm_gru}

Classification Report:
{classification_report(y_test, y_pred_gru, target_names=['No Block', 'Block Present'])}

{'=' * 80}
BEST MODEL: {'1D CNN' if test_acc_1d_cnn >= test_acc_gru else 'GRU'}
{'=' * 80}
"""

summary_path = os.path.join(base_dir, f"TRAINING_SUMMARY_{timestamp}.txt")
with open(summary_path, 'w') as f:
    f.write(summary)
print(f"✓ Summary report saved to {summary_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nResults saved in: {base_dir}")
print(f"  - {cnn_dir}/")
print(f"  - {gru_dir}/")
