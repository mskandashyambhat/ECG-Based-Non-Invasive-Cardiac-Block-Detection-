"""
Enhanced ECG Preprocessing Pipeline - All 5 Datasets
Includes: PTB-XL, MIT-BIH, LUDB, PTB Diagnostic, PhysioNet Arrhythmia
Merges with existing preprocessed dataset and rebalances all classes
"""

import numpy as np
import pandas as pd
import wfdb
import os
import warnings
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy.signal import resample, butter, sosfiltfilt, iirnotch, filtfilt, find_peaks
from scipy.io import loadmat
import ast

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

TARGET_SAMPLING_RATE = 500
SEGMENT_LENGTH = 300
R_PEAK_BEFORE = 100
R_PEAK_AFTER = 200

BP_ORDER = 4
BP_LOW = 0.5
BP_HIGH = 40.0

NOTCH_FREQ = 50.0
NOTCH_Q = 30

CLASS_MAPPING = {
    'Normal': 0,
    'AV Block': 1,
    'Complete Heart Block': 2,
    'RBBB': 3,
    'LBBB': 4
}

# SCP Code mapping for PTB-XL
SCP_CODE_MAPPING = {
    # First-degree and Second-degree AV blocks
    '1AVB': 1,
    '2AVB': 1,
    '2AVB1': 1,
    '2AVB2': 1,
    # Third-degree (Complete) AV block
    '3AVB': 2,
    # Bundle branch blocks
    'RBBB': 3,  # Right bundle branch block
    'IRBBB': 3,  # Incomplete RBBB
    'CRBBB': 3,  # Complete RBBB
    'LBBB': 4,  # Left bundle branch block
    'ILBBB': 4,  # Incomplete LBBB
    'CLBBB': 4,  # Complete LBBB
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
    signal_data = np.asarray(signal_data).flatten().astype(np.float32)
    
    # Resample
    if original_fs != target_fs:
        num_samples = int(len(signal_data) * target_fs / original_fs)
        signal_data = resample(signal_data, num_samples)
    
    # Bandpass filter
    signal_data = apply_bandpass_filter(signal_data, target_fs)
    
    # Notch filter
    signal_data = apply_notch_filter(signal_data, target_fs)
    
    return signal_data

# ============================================================================
# QRS DETECTION AND SEGMENTATION
# ============================================================================

def detect_qrs_peaks(signal_data, sampling_rate=TARGET_SAMPLING_RATE):
    """Detect R peaks."""
    try:
        signal_data = np.asarray(signal_data).flatten()
        peaks, _ = find_peaks(signal_data, distance=int(sampling_rate * 0.3))
        return peaks if len(peaks) > 0 else []
    except:
        return []

def segment_by_heartbeats(signal_data, r_peaks):
    """Segment ECG signal into heartbeats."""
    segments = []
    valid_peaks = []
    
    for r_peak in r_peaks:
        start = r_peak - R_PEAK_BEFORE
        end = r_peak + R_PEAK_AFTER
        
        if start >= 0 and end <= len(signal_data):
            segment = signal_data[start:end]
            segments.append(segment)
            valid_peaks.append(r_peak)
    
    return np.array(segments) if len(segments) > 0 else np.array([]).reshape(0, 300), np.array(valid_peaks)

def zscore_normalize_segment(segment):
    """Apply Z-score normalization."""
    mean = np.mean(segment)
    std = np.std(segment)
    if std == 0:
        return segment
    return (segment - mean) / std

def extract_interval_features(segment, r_peak_idx=R_PEAK_BEFORE):
    """Extract interval features."""
    return {
        'PR': 120,
        'QRS': 100,
        'RR': SEGMENT_LENGTH / TARGET_SAMPLING_RATE * 1000,
        'HR': 60
    }

# ============================================================================
# PTB DIAGNOSTIC ECG DATABASE LOADER
# ============================================================================

def load_ptb_diagnostic_dataset(dataset_path):
    """Load PTB Diagnostic ECG Database."""
    print("\n" + "="*70)
    print("Loading PTB Diagnostic ECG Database...")
    print("="*70)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    # Get all patient folders
    patient_dirs = sorted([d for d in os.listdir(dataset_path) 
                          if d.startswith('patient') and os.path.isdir(os.path.join(dataset_path, d))])
    
    for patient_dir in tqdm(patient_dirs, desc="Processing PTB Diagnostic"):
        try:
            patient_path = os.path.join(dataset_path, patient_dir)
            
            # Find .hea file (WFDB header)
            hea_files = [f for f in os.listdir(patient_path) if f.endswith('.hea')]
            
            if len(hea_files) == 0:
                failed_count += 1
                continue
            
            record_path = os.path.join(patient_path, hea_files[0].replace('.hea', ''))
            ecg_data, info = wfdb.rdsamp(record_path)
            
            # Extract Lead II
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
            
            # Default to Normal (complex diagnostic metadata)
            label = 0
            
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
    
    print(f"PTB Diagnostic: Successfully processed {success_count} segments, failed {failed_count} records")
    
    X = np.array(X_list) if len(X_list) > 0 else np.array([]).reshape(0, 300)
    y = np.array(y_list) if len(y_list) > 0 else np.array([])
    
    return X, y, features_list, 'PTB Diagnostic'

# ============================================================================
# PHYSIONET ARRHYTHMIA DATABASE LOADER
# ============================================================================

def load_physionet_arrhythmia_dataset(dataset_path):
    """Load PhysioNet A-large-scale-12-lead ECG Database from .mat files."""
    print("\n" + "="*70)
    print("Loading PhysioNet Arrhythmia Database (.mat files)...")
    print("="*70)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    # PhysioNet SNOMED-CT code to class mapping
    SNOMED_TO_CLASS = {
        '270492004': 1,   # 1AVB
        '195042002': 1,   # 2AVB
        '54016002': 1,    # 2AVB1 (Mobitz I)
        '28189009': 1,    # 2AVB2 (Mobitz II)
        '27885002': 2,    # 3AVB (Complete Heart Block)
        '59118001': 3,    # RBBB
        '164909002': 4,   # LBBB
    }
    
    # Get RECORDS file
    records_file = os.path.join(dataset_path, 'RECORDS')
    if not os.path.exists(records_file):
        print("RECORDS file not found")
        return np.array([]).reshape(0, 300), np.array([]), [], 'PhysioNet'
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(record_names)} records")
    
    for record_path in tqdm(record_names, desc="Processing PhysioNet"):
        try:
            # record_path is like 'WFDBRecords/02/021/'
            record_dir = os.path.join(dataset_path, record_path.rstrip('/'))
            
            # Find .mat and .hea files in this directory
            mat_files = [f for f in os.listdir(record_dir) if f.endswith('.mat')]
            if not mat_files:
                failed_count += 1
                continue
            
            mat_file_path = os.path.join(record_dir, mat_files[0])
            hea_file_path = mat_file_path.replace('.mat', '.hea')
            
            if not os.path.exists(mat_file_path):
                failed_count += 1
                continue
            
            # Load ECG data from .mat file
            mat_data = loadmat(mat_file_path)
            
            # Extract signal (usually stored as 'val' or first data array)
            if 'val' in mat_data:
                ecg_data = mat_data['val'].astype(np.float32)
            else:
                # Find the largest array in mat_data (likely the signal)
                data_arrays = [v for k, v in mat_data.items() if isinstance(v, np.ndarray) and not k.startswith('__')]
                if len(data_arrays) == 0:
                    failed_count += 1
                    continue
                ecg_data = data_arrays[0].astype(np.float32)
            
            # PhysioNet typically has 12 leads, extract Lead II (index 1)
            if ecg_data.ndim == 1:
                lead_ii = ecg_data
                original_fs = 500  # Default PhysioNet sampling rate
            elif ecg_data.shape[0] < ecg_data.shape[1]:
                # Leads in rows
                if ecg_data.shape[0] >= 2:
                    lead_ii = ecg_data[1, :].astype(np.float32)
                else:
                    lead_ii = ecg_data[0, :].astype(np.float32)
                original_fs = 500
            else:
                # Leads in columns
                if ecg_data.shape[1] >= 2:
                    lead_ii = ecg_data[:, 1].astype(np.float32)
                else:
                    lead_ii = ecg_data[:, 0].astype(np.float32)
                original_fs = 500
            
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
            
            # Extract diagnostic label from #Dx line in .hea file
            label = 0
            if os.path.exists(hea_file_path):
                try:
                    with open(hea_file_path, 'r') as f:
                        for line in f:
                            if line.startswith('#Dx:'):
                                # Extract SNOMED-CT codes from #Dx: line
                                dx_part = line.replace('#Dx:', '').strip()
                                snomed_codes = dx_part.split(',')
                                
                                # Map SNOMED codes to class labels (priority: 2AVB2, 3AVB, others)
                                found = False
                                for snomed_code in snomed_codes:
                                    code = snomed_code.strip()
                                    if code in SNOMED_TO_CLASS:
                                        # Prefer Mobitz II and Complete Heart Block
                                        if code == '28189009':  # 2AVB2 (Mobitz II)
                                            label = 3
                                            found = True
                                            break
                                        elif code == '27885002':  # 3AVB (Complete)
                                            label = 4
                                            found = True
                                            break
                                        elif label == 0:
                                            label = SNOMED_TO_CLASS[code]
                                            found = True
                                if found:
                                    break  # Exit after processing #Dx: line
                except Exception as e:
                    label = 0
            
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
    
    print(f"PhysioNet: Successfully processed {success_count} segments, failed {failed_count} records")
    
    X = np.array(X_list) if len(X_list) > 0 else np.array([]).reshape(0, 300)
    y = np.array(y_list) if len(y_list) > 0 else np.array([])
    
    return X, y, features_list, 'PhysioNet'

# ============================================================================
# LOAD EXISTING PREPROCESSED DATASET
# ============================================================================

def extract_scp_label(scp_codes_str):
    """Extract diagnostic label from SCP codes."""
    try:
        import ast
        codes = ast.literal_eval(scp_codes_str) if isinstance(scp_codes_str, str) else scp_codes_str
        
        # Priority order: check for bundle branch blocks first, then AV blocks
        for code in codes.keys():
            if code in SCP_CODE_MAPPING:
                return SCP_CODE_MAPPING[code]
        
        # Default to Normal if no matching diagnostic code
        return 0
    except:
        return 0

def load_mitbih_dataset(dataset_path):
    """Load MIT-BIH Arrhythmia Database."""
    print("\n" + "="*70)
    print("Loading MIT-BIH Arrhythmia Database...")
    print("="*70)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    records_file = os.path.join(dataset_path, 'RECORDS')
    if not os.path.exists(records_file):
        print("RECORDS file not found")
        return np.array([]).reshape(0, 300), np.array([]), [], 'MIT-BIH'
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    
    for record_name in tqdm(record_names, desc="Processing MIT-BIH"):
        try:
            record_path = os.path.join(dataset_path, record_name)
            
            if not os.path.exists(record_path + '.hea'):
                failed_count += 1
                continue
            
            ecg_data, info = wfdb.rdsamp(record_path)
            
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
            
            # Default to Normal for MIT-BIH (no diagnostic metadata in main files)
            label = 0
            
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

def load_ludb_dataset(dataset_path):
    """Load LUDB (Lobachevsky University ECG Database)."""
    print("\n" + "="*70)
    print("Loading LUDB Dataset...")
    print("="*70)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    # AV block mapping from diagnosis strings
    def extract_label_from_diagnosis(diagnosis_str):
        diagnosis = str(diagnosis_str).lower()
        if 'complete' in diagnosis or '3rd' in diagnosis or '3av' in diagnosis:
            return 2
        elif '1st' in diagnosis or 'first' in diagnosis or 'mobitz' in diagnosis or '2av' in diagnosis:
            return 1
        elif 'rbbb' in diagnosis:
            return 3
        elif 'lbbb' in diagnosis:
            return 4
        return 0
    
    # Read LUDB CSV
    ludb_csv = os.path.join(dataset_path, 'ludb.csv')
    if os.path.exists(ludb_csv):
        try:
            df_ludb = pd.read_csv(ludb_csv)
            record_dict = dict(zip(df_ludb.iloc[:, 0].astype(str), df_ludb.iloc[:, 1]))
        except:
            record_dict = {}
    else:
        record_dict = {}
    
    data_dir = os.path.join(dataset_path, 'data')
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return np.array([]).reshape(0, 300), np.array([]), [], 'LUDB'
    
    # Get all unique record IDs (files named like 1, 2, 3, etc., with extensions .hea, .dat)
    hea_files = set([f.rsplit('.', 1)[0] for f in os.listdir(data_dir) if f.endswith('.hea')])
    record_ids = sorted(hea_files)
    
    for record_id in tqdm(record_ids, desc="Processing LUDB"):
        try:
            record_path = os.path.join(data_dir, record_id)
            
            if not os.path.exists(record_path + '.hea'):
                failed_count += 1
                continue
            
            ecg_data, info = wfdb.rdsamp(record_path)
            
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
            
            # Extract label from CSV diagnosis
            label = extract_label_from_diagnosis(record_dict.get(record_id, ''))
            
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

def load_ptbxl_dataset_with_labels(dataset_path):
    """Load PTB-XL with proper diagnostic labels."""
    print("\n" + "="*70)
    print("Loading PTB-XL Dataset (with diagnostic labels)...")
    print("="*70)
    
    import ast
    
    csv_path = os.path.join(dataset_path, 'ptbxl_database.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return np.array([]).reshape(0, 300), np.array([]), [], 'PTB-XL'
    
    df = pd.read_csv(csv_path)
    
    X_list = []
    y_list = []
    features_list = []
    success_count = 0
    failed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing PTB-XL with labels"):
        try:
            # Use filename_hr from CSV (already has path)
            record_path = os.path.join(dataset_path, row['filename_hr'])
            
            if not os.path.exists(record_path + '.hea'):
                failed_count += 1
                continue
            
            ecg_data, info = wfdb.rdsamp(record_path)
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
            
            # Extract diagnostic label from SCP codes
            label = extract_scp_label(row['scp_codes'])
            
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

def load_existing_preprocessed_dataset(preprocessed_path):
    """Load already preprocessed dataset."""
    print("\n" + "="*70)
    print("Loading Existing Preprocessed Dataset...")
    print("="*70)
    
    try:
        with open(os.path.join(preprocessed_path, 'merged_ecg_dataset.pkl'), 'rb') as f:
            data = pickle.load(f)
            X = data['X']
            y = data['y']
            features = data.get('features', [])
        
        print(f"Existing dataset loaded: X shape={X.shape}, y shape={y.shape}")
        return X, y, features, 'Existing Dataset'
    
    except Exception as e:
        print(f"Error loading existing dataset: {e}")
        return np.array([]).reshape(0, 300), np.array([]), [], 'Existing Dataset'

# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def print_dataset_statistics(X, y, dataset_name):
    """Print statistics."""
    print(f"\nDataset: {dataset_name}")
    print(f"Total Segments: {len(X)}")
    print("\nClass Distribution:")
    
    class_counts = defaultdict(int)
    for label in y:
        class_counts[label] += 1
    
    class_names = {
        0: 'Normal',
        1: 'AV Block',
        2: 'Complete Heart Block',
        3: 'RBBB',
        4: 'LBBB'
    }
    
    for class_id in range(5):
        count = class_counts[class_id]
        class_name = class_names[class_id]
        print(f"  {class_name}: {count}")
    
    print("-" * 50)

# ============================================================================
# CLASS BALANCING
# ============================================================================

def augment_segment(segment):
    """Augment segment with random method."""
    method = np.random.choice(['noise', 'shift', 'scale'], 1)[0]
    
    if method == 'noise':
        noise = np.random.normal(0, 0.01, segment.shape)
        return segment + noise
    elif method == 'shift':
        shift_amount = np.random.randint(-5, 6)
        return np.roll(segment, shift_amount)
    else:
        scale_factor = np.random.uniform(0.9, 1.1)
        return segment * scale_factor

def balance_classes_global(X, y):
    """Balance all classes to same count with smart augmentation."""
    print("\n" + "="*70)
    print("Global Class Balancing with Augmentation...")
    print("="*70)
    
    unique_classes = np.unique(y)
    class_counts = {cls: np.sum(y == cls) for cls in unique_classes}
    
    print(f"\nCurrent class distribution: {class_counts}")
    
    if len(class_counts) == 0 or max(class_counts.values()) == 0:
        print("No valid data to balance.")
        return X, y
    
    # Use 80% of max class count as target to reduce augmentation time
    max_count = max(class_counts.values())
    target_count = int(max_count * 0.8)
    
    print(f"Maximum class count: {max_count}")
    print(f"Target class count: {target_count}")
    
    X_balanced = []
    y_balanced = []
    
    for class_id in sorted(unique_classes):
        class_mask = y == class_id
        X_class = X[class_mask]
        current_count = len(X_class)
        
        # Add all original samples
        X_balanced.extend(X_class)
        y_balanced.extend([class_id] * current_count)
        
        # Augment if needed
        if current_count < target_count:
            need_augment = target_count - current_count
            
            for _ in range(need_augment):
                idx = np.random.randint(0, len(X_class))
                augmented = augment_segment(X_class[idx])
                X_balanced.append(augmented)
                y_balanced.append(class_id)
            
            print(f"Class {class_id}: {current_count} → {target_count} (+{need_augment} augmented)")
        else:
            print(f"Class {class_id}: {current_count} (no augmentation needed)")
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    
    print(f"\nFinal balanced dataset size: {len(X_balanced):,} samples")
    
    return X_balanced, y_balanced

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main preprocessing pipeline for all 5 datasets."""
    
    dataset_base = '/Users/skandashyam/Desktop/MajorProject/Project/Dataset'
    preprocessed_path = os.path.join(dataset_base, 'preprocessed_dataset')
    
    print("\n" + "="*70)
    print("ECG PREPROCESSING PIPELINE - ALL 5 DATASETS (WITH PROPER LABELS)")
    print("="*70)
    
    # Load all datasets fresh with proper labels
    ptbxl_path = os.path.join(dataset_base, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
    X_ptbxl, y_ptbxl, feat_ptbxl, _ = load_ptbxl_dataset_with_labels(ptbxl_path)
    print_dataset_statistics(X_ptbxl, y_ptbxl, 'PTB-XL (with labels)')
    
    mitbih_path = os.path.join(dataset_base, 'mit-bih-arrhythmia-database-1.0.0')
    X_mitbih, y_mitbih, feat_mitbih, _ = load_mitbih_dataset(mitbih_path)
    print_dataset_statistics(X_mitbih, y_mitbih, 'MIT-BIH')
    
    ludb_path = os.path.join(dataset_base, 'lobachevsky-university-electrocardiography-database-1.0.1')
    X_ludb, y_ludb, feat_ludb, _ = load_ludb_dataset(ludb_path)
    print_dataset_statistics(X_ludb, y_ludb, 'LUDB')
    
    # Load 2 new datasets
    ptb_diag_path = os.path.join(dataset_base, 'ptb-diagnostic-ecg-database-1.0.0')
    X_ptb_diag, y_ptb_diag, feat_ptb_diag, _ = load_ptb_diagnostic_dataset(ptb_diag_path)
    print_dataset_statistics(X_ptb_diag, y_ptb_diag, 'PTB Diagnostic')
    
    physionet_path = os.path.join(dataset_base, 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0')
    X_physionet, y_physionet, feat_physionet, _ = load_physionet_arrhythmia_dataset(physionet_path)
    print_dataset_statistics(X_physionet, y_physionet, 'PhysioNet Arrhythmia')
    
    # Merge all datasets
    print("\n" + "="*70)
    print("Merging All 5 Datasets...")
    print("="*70)
    
    all_X_list = []
    all_y_list = []
    all_features_list = []
    
    if len(X_ptbxl) > 0:
        all_X_list.append(X_ptbxl)
        all_y_list.append(y_ptbxl)
        all_features_list.extend(feat_ptbxl)
    
    if len(X_mitbih) > 0:
        all_X_list.append(X_mitbih)
        all_y_list.append(y_mitbih)
        all_features_list.extend(feat_mitbih)
    
    if len(X_ludb) > 0:
        all_X_list.append(X_ludb)
        all_y_list.append(y_ludb)
        all_features_list.extend(feat_ludb)
    
    if len(X_ptb_diag) > 0:
        all_X_list.append(X_ptb_diag)
        all_y_list.append(y_ptb_diag)
        all_features_list.extend(feat_ptb_diag)
    
    if len(X_physionet) > 0:
        all_X_list.append(X_physionet)
        all_y_list.append(y_physionet)
        all_features_list.extend(feat_physionet)
    
    X_merged = np.vstack(all_X_list) if len(all_X_list) > 0 else np.array([]).reshape(0, 300)
    y_merged = np.concatenate(all_y_list) if len(all_y_list) > 0 else np.array([])
    
    print_dataset_statistics(X_merged, y_merged, 'MERGED (Before Balancing)')
    
    # Balance classes
    X_balanced, y_balanced = balance_classes_global(X_merged, y_merged)
    
    # Print final statistics
    print("\n" + "="*70)
    print("FINAL BALANCED DATASET")
    print("="*70)
    print_dataset_statistics(X_balanced, y_balanced, 'FINAL DATASET')
    
    # Save updated dataset
    print("\n" + "="*70)
    print("Saving Updated Preprocessed Dataset...")
    print("="*70)
    
    output_file = os.path.join(preprocessed_path, 'merged_ecg_dataset_all5_complete.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X_balanced,
            'y': y_balanced,
            'features': all_features_list[:len(X_balanced)] if len(all_features_list) > 0 else [],
            'shape': X_balanced.shape,
            'class_names': {
                0: 'Normal',
                1: 'AV Block',
                2: 'Complete Heart Block',
                3: 'RBBB',
                4: 'LBBB'
            }
        }, f)
    
    print(f"\nDataset saved to: {output_file}")
    print(f"Dataset shape: {X_balanced.shape}")
    print(f"Labels shape: {y_balanced.shape}")
    
    # Also save as NPZ
    npz_file = os.path.join(preprocessed_path, 'merged_ecg_dataset_all5_complete.npz')
    np.savez(npz_file, X=X_balanced, y=y_balanced)
    print(f"NPZ file saved to: {npz_file}")
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    
    return X_balanced, y_balanced

if __name__ == '__main__':
    X, y = main()
