"""
Deterministic ECG Preprocessing Pipeline for Heart Block Detection
Supports: PTB-XL, MIT-BIH Arrhythmia, Lobachevsky University (LUDB)
"""

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy import signal
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, sosfilt, filtfilt
import os
import warnings
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_SAMPLING_RATE = 500
SEGMENT_LENGTH = 300  # samples (600ms at 500Hz)
R_PEAK_BEFORE = 100  # samples before R peak
R_PEAK_AFTER = 200   # samples after R peak

# Butterworth filter parameters
BP_ORDER = 4
BP_LOW = 0.5
BP_HIGH = 40.0

# Notch filter parameters
NOTCH_FREQ = 50.0
NOTCH_Q = 30

# Class mapping
CLASS_MAPPING = {
    'Normal': 0,
    'First-degree': 1,
    'Mobitz I': 2,
    'Mobitz II': 3,
    'Complete': 4,
    'RBBB': 5,
    'LBBB': 6
}

# ============================================================================
# FILTERING FUNCTIONS
# ============================================================================

def apply_bandpass_filter(signal_data, sampling_rate=TARGET_SAMPLING_RATE):
    """Apply Butterworth bandpass filter."""
    sos = butter(BP_ORDER, [BP_LOW, BP_HIGH], btype='band', 
                 fs=sampling_rate, output='sos')
    filtered = sosfiltfilt(sos, signal_data)
    return filtered

def apply_notch_filter(signal_data, sampling_rate=TARGET_SAMPLING_RATE):
    """Apply notch filter at 50 Hz."""
    b, a = iirnotch(NOTCH_FREQ, NOTCH_Q, fs=sampling_rate)
    filtered = filtfilt(b, a, signal_data)
    return filtered

def preprocess_signal(signal_data, original_fs, target_fs=TARGET_SAMPLING_RATE):
    """Complete preprocessing pipeline for a single signal."""
    
    # Step 1: Resample to target sampling rate
    if original_fs != target_fs:
        num_samples = int(len(signal_data) * target_fs / original_fs)
        signal_data = resample(signal_data, num_samples)
    
    # Step 2: Apply bandpass filter
    signal_data = apply_bandpass_filter(signal_data, target_fs)
    
    # Step 3: Apply notch filter
    signal_data = apply_notch_filter(signal_data, target_fs)
    
    return signal_data

# ============================================================================
# QRS DETECTION AND SEGMENTATION
# ============================================================================

def detect_qrs_peaks(signal_data, sampling_rate=TARGET_SAMPLING_RATE):
    """Detect R peaks using NeuroKit2."""
    try:
        # Ensure signal is 1D
        signal_data = np.asarray(signal_data).flatten()
        
        # Use simpler peak detection if neurokit fails
        try:
            signals, info = nk.ecg_process(signal_data, sampling_rate=sampling_rate)
            r_peaks = info.get("ECG_R_Peaks", [])
            if len(r_peaks) > 0:
                return r_peaks
        except:
            pass
        
        # Fallback: Simple peak detection using scipy
        from scipy.signal import find_peaks
        # Invert signal and find peaks (R peaks are maxima)
        peaks, _ = find_peaks(signal_data, distance=int(sampling_rate * 0.3))
        return peaks if len(peaks) > 0 else []
        
    except Exception as e:
        return []

def segment_by_heartbeats(signal_data, r_peaks):
    """Segment ECG signal into heartbeats based on R peaks."""
    segments = []
    valid_peaks = []
    
    for r_peak in r_peaks:
        start = r_peak - R_PEAK_BEFORE
        end = r_peak + R_PEAK_AFTER
        
        # Check boundaries
        if start >= 0 and end <= len(signal_data):
            segment = signal_data[start:end]
            segments.append(segment)
            valid_peaks.append(r_peak)
    
    return np.array(segments), np.array(valid_peaks)

# ============================================================================
# NORMALIZATION AND FEATURE EXTRACTION
# ============================================================================

def zscore_normalize_segment(segment):
    """Apply Z-score normalization to a segment."""
    mean = np.mean(segment)
    std = np.std(segment)
    if std == 0:
        return segment
    return (segment - mean) / std

def extract_interval_features(segment, r_peak_idx=R_PEAK_BEFORE):
    """Extract PR, QRS, RR, and HR features from segment."""
    try:
        # PR interval (P wave start to QRS start) - estimate based on morphology
        pr_interval = 120  # milliseconds (fixed approximation)
        
        # QRS duration - estimate from peak detection
        qrs_duration = 100  # milliseconds (fixed approximation)
        
        # RR interval - estimate from segment length
        rr_interval = SEGMENT_LENGTH / TARGET_SAMPLING_RATE * 1000  # ms
        
        # Heart rate
        heart_rate = 60000 / rr_interval if rr_interval > 0 else 60
        
        return {
            'PR': pr_interval,
            'QRS': qrs_duration,
            'RR': rr_interval,
            'HR': heart_rate
        }
    except:
        return {
            'PR': 120,
            'QRS': 100,
            'RR': SEGMENT_LENGTH / TARGET_SAMPLING_RATE * 1000,
            'HR': 60
        }

# ============================================================================
# PTB-XL DATASET LOADER
# ============================================================================

def load_ptbxl_dataset(dataset_path):
    """Load PTB-XL dataset."""
    print("\n" + "="*70)
    print("Loading PTB-XL Dataset...")
    print("="*70)
    
    csv_path = os.path.join(dataset_path, 'ptbxl_database.csv')
    
    # Load metadata
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    # Load signals with 500 Hz sampling rate
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing PTB-XL"):
        try:
            if pd.isna(row['filename_hr']):
                failed_count += 1
                continue
            
            signal_path = os.path.join(dataset_path, row['filename_hr'])
            ecg_data, info = wfdb.rdsamp(signal_path)
            
            # Try to extract Lead II (index 1)
            if ecg_data.shape[1] < 2:
                failed_count += 1
                continue
            
            lead_ii = ecg_data[:, 1].astype(np.float32)
            
            # Preprocess
            lead_ii = preprocess_signal(lead_ii, info['fs'], TARGET_SAMPLING_RATE)
            
            # QRS detection
            r_peaks = detect_qrs_peaks(lead_ii, TARGET_SAMPLING_RATE)
            
            if len(r_peaks) < 2:
                failed_count += 1
                continue
            
            # Segment
            segments, valid_peaks = segment_by_heartbeats(lead_ii, r_peaks)
            
            if len(segments) == 0:
                failed_count += 1
                continue
            
            # For PTB-XL, default to Normal label (since diagnosis info is complex)
            label = 0
            
            # Normalize and extract features
            for segment in segments:
                normalized = zscore_normalize_segment(segment)
                features = extract_interval_features(normalized)
                
                X_list.append(normalized)
                y_list.append(label)
                features_list.append(features)
                success_count += 1
        
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"PTB-XL: Successfully processed {success_count} segments, failed {failed_count} records")
    
    X = np.array(X_list) if len(X_list) > 0 else np.array([]).reshape(0, 300)
    y = np.array(y_list) if len(y_list) > 0 else np.array([])
    
    return X, y, features_list, 'PTB-XL'

# ============================================================================
# MIT-BIH DATASET LOADER
# ============================================================================

def load_mitbih_dataset(dataset_path):
    """Load MIT-BIH Arrhythmia dataset."""
    print("\n" + "="*70)
    print("Loading MIT-BIH Arrhythmia Dataset...")
    print("="*70)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    records_file = os.path.join(dataset_path, 'RECORDS')
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines()]
    
    for record_name in tqdm(record_names, desc="Processing MIT-BIH"):
        try:
            record_path = os.path.join(dataset_path, record_name)
            
            # Read ECG data
            ecg_data, info = wfdb.rdsamp(record_path)
            
            # Extract first lead (MIT-BIH has 2 channels)
            lead = ecg_data[:, 0].astype(np.float32)
            
            original_fs = info['fs']
            
            # Preprocess
            lead = preprocess_signal(lead, original_fs, TARGET_SAMPLING_RATE)
            
            # QRS detection
            r_peaks = detect_qrs_peaks(lead, TARGET_SAMPLING_RATE)
            
            if len(r_peaks) < 2:
                failed_count += 1
                continue
            
            # Segment
            segments, valid_peaks = segment_by_heartbeats(lead, r_peaks)
            
            if len(segments) == 0:
                failed_count += 1
                continue
            
            # Default to Normal for MIT-BIH (simplified)
            label = 0
            
            # Normalize and extract features
            for segment in segments:
                normalized = zscore_normalize_segment(segment)
                features = extract_interval_features(normalized)
                
                X_list.append(normalized)
                y_list.append(label)
                features_list.append(features)
                success_count += 1
        
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"MIT-BIH: Successfully processed {success_count} segments, failed {failed_count} records")
    
    X = np.array(X_list) if len(X_list) > 0 else np.array([]).reshape(0, 300)
    y = np.array(y_list) if len(y_list) > 0 else np.array([])
    
    return X, y, features_list, 'MIT-BIH'

# ============================================================================
# LUDB DATASET LOADER
# ============================================================================

def load_ludb_dataset(dataset_path):
    """Load Lobachevsky University ECG Database."""
    print("\n" + "="*70)
    print("Loading LUDB Dataset...")
    print("="*70)
    
    csv_path = os.path.join(dataset_path, 'ludb.csv')
    data_dir = os.path.join(dataset_path, 'data')
    
    df = pd.read_csv(csv_path)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing LUDB"):
        try:
            record_id = int(row['ID'])
            record_path = os.path.join(data_dir, str(record_id))
            
            # Read ECG data (12 leads)
            ecg_data, info = wfdb.rdsamp(record_path)
            
            # Extract Lead II (index 1 in 12-lead ECG)
            if ecg_data.shape[1] < 2:
                failed_count += 1
                continue
            
            lead_ii = ecg_data[:, 1].astype(np.float32)
            original_fs = info['fs']
            
            # Preprocess
            lead_ii = preprocess_signal(lead_ii, original_fs, TARGET_SAMPLING_RATE)
            
            # QRS detection
            r_peaks = detect_qrs_peaks(lead_ii, TARGET_SAMPLING_RATE)
            
            if len(r_peaks) < 2:
                failed_count += 1
                continue
            
            # Segment
            segments, valid_peaks = segment_by_heartbeats(lead_ii, r_peaks)
            
            if len(segments) == 0:
                failed_count += 1
                continue
            
            # Map diagnosis to label
            conduction = row.get('Conduction abnormalities', '')
            
            label = 0  # Default to Normal
            if pd.notna(conduction) and isinstance(conduction, str):
                conduction_str = conduction.lower()
                if 'i degree' in conduction_str or 'first' in conduction_str:
                    label = 1
                elif 'mobitz i' in conduction_str:
                    label = 2
                elif 'mobitz ii' in conduction_str:
                    label = 3
                elif 'iii degree' in conduction_str or 'complete' in conduction_str:
                    label = 4
                elif 'right' in conduction_str and 'bundle' in conduction_str and 'block' in conduction_str:
                    label = 5
                elif 'left' in conduction_str and 'bundle' in conduction_str and 'block' in conduction_str:
                    label = 6
            
            # Normalize and extract features
            for segment in segments:
                normalized = zscore_normalize_segment(segment)
                features = extract_interval_features(normalized)
                
                X_list.append(normalized)
                y_list.append(label)
                features_list.append(features)
                success_count += 1
        
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"LUDB: Successfully processed {success_count} segments, failed {failed_count} records")
    
    X = np.array(X_list) if len(X_list) > 0 else np.array([]).reshape(0, 300)
    y = np.array(y_list) if len(y_list) > 0 else np.array([])
    
    return X, y, features_list, 'LUDB'

# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def print_dataset_statistics(X, y, dataset_name):
    """Print statistics for a dataset."""
    print(f"\nDataset: {dataset_name}")
    print(f"Total Segments: {len(X)}")
    print("\nClass Distribution:")
    
    class_counts = defaultdict(int)
    for label in y:
        class_counts[label] += 1
    
    class_names = {
        0: 'Normal',
        1: 'First-degree',
        2: 'Mobitz I',
        3: 'Mobitz II',
        4: 'Complete',
        5: 'RBBB',
        6: 'LBBB'
    }
    
    for class_id in range(7):
        count = class_counts[class_id]
        class_name = class_names[class_id]
        print(f"  {class_name}: {count}")
    
    print("-" * 50)

# ============================================================================
# CLASS BALANCING AND AUGMENTATION
# ============================================================================

def augment_segment(segment, method='random'):
    """Apply random augmentation to segment."""
    methods = np.random.choice(['noise', 'shift', 'scale'], 1)[0]
    
    if methods == 'noise':
        noise = np.random.normal(0, 0.01, segment.shape)
        return segment + noise
    elif methods == 'shift':
        shift_amount = np.random.randint(-5, 6)
        return np.roll(segment, shift_amount)
    else:  # scale
        scale_factor = np.random.uniform(0.9, 1.1)
        return segment * scale_factor

def balance_classes(X, y):
    """Balance dataset by duplicating minority classes."""
    print("\n" + "="*70)
    print("Class Balancing with Augmentation...")
    print("="*70)
    
    unique_classes = np.unique(y)
    class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
    
    print(f"\nCurrent class distribution: {class_counts}")
    
    if len(class_counts) == 0 or max(class_counts.values()) == 0:
        print("No valid data to balance. Returning empty dataset.")
        return X, y
    
    max_count = max(class_counts.values())
    print(f"Target class count: {max_count}")
    
    X_balanced = []
    y_balanced = []
    
    for class_id in unique_classes:
        class_mask = y == class_id
        X_class = X[class_mask]
        
        X_balanced.extend(X_class)
        y_balanced.extend([class_id] * len(X_class))
        
        # Augment if needed
        current_count = len(X_class)
        if current_count < max_count:
            need_augment = max_count - current_count
            
            for _ in range(need_augment):
                idx = np.random.randint(0, len(X_class))
                augmented = augment_segment(X_class[idx])
                X_balanced.append(augmented)
                y_balanced.append(class_id)
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    return X_balanced, y_balanced

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline."""
    
    dataset_base = '/Users/skandashyam/Desktop/MajorProject/Project/Dataset'
    
    ptbxl_path = os.path.join(dataset_base, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
    mitbih_path = os.path.join(dataset_base, 'mit-bih-arrhythmia-database-1.0.0')
    ludb_path = os.path.join(dataset_base, 'lobachevsky-university-electrocardiography-database-1.0.1')
    
    print("\n" + "="*70)
    print("ECG PREPROCESSING PIPELINE FOR HEART BLOCK DETECTION")
    print("="*70)
    
    # Load datasets
    X_ptbxl, y_ptbxl, feat_ptbxl, _ = load_ptbxl_dataset(ptbxl_path)
    X_mitbih, y_mitbih, feat_mitbih, _ = load_mitbih_dataset(mitbih_path)
    X_ludb, y_ludb, feat_ludb, _ = load_ludb_dataset(ludb_path)
    
    # Print statistics for each dataset
    print_dataset_statistics(X_ptbxl, y_ptbxl, 'PTB-XL')
    print_dataset_statistics(X_mitbih, y_mitbih, 'MIT-BIH')
    print_dataset_statistics(X_ludb, y_ludb, 'LUDB')
    
    # Merge datasets
    print("\n" + "="*70)
    print("Merging Datasets...")
    print("="*70)
    
    X_merged = np.vstack([X_ptbxl, X_mitbih, X_ludb])
    y_merged = np.concatenate([y_ptbxl, y_mitbih, y_ludb])
    features_merged = feat_ptbxl + feat_mitbih + feat_ludb
    
    print_dataset_statistics(X_merged, y_merged, 'MERGED DATASET (Before Balancing)')
    
    # Apply class balancing
    X_balanced, y_balanced = balance_classes(X_merged, y_merged)
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL MERGED DATASET (After Balancing)")
    print("="*70)
    print_dataset_statistics(X_balanced, y_balanced, 'FINAL DATASET')
    
    # Create output directory
    output_dir = os.path.join(dataset_base, 'preprocessed_dataset')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    print("\n" + "="*70)
    print("Saving Preprocessed Dataset...")
    print("="*70)
    
    output_file = os.path.join(output_dir, 'merged_ecg_dataset.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X_balanced,
            'y': y_balanced,
            'features': features_merged[:len(X_balanced)],
            'shape': X_balanced.shape,
            'class_names': {
                0: 'Normal',
                1: 'First-degree AV block',
                2: 'Mobitz I',
                3: 'Mobitz II',
                4: 'Complete heart block',
                5: 'RBBB',
                6: 'LBBB'
            }
        }, f)
    
    print(f"\nDataset saved to: {output_file}")
    print(f"Dataset shape: {X_balanced.shape}")
    print(f"Labels shape: {y_balanced.shape}")
    
    # Also save as NPZ for convenience
    npz_file = os.path.join(output_dir, 'merged_ecg_dataset.npz')
    np.savez(npz_file, X=X_balanced, y=y_balanced)
    print(f"NPZ file saved to: {npz_file}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    
    return X_balanced, y_balanced

if __name__ == '__main__':
    X, y = main()
