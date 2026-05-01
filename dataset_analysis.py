"""
Dataset Analysis Script
Provides comprehensive breakdown of all datasets and their class distributions
"""

import numpy as np
import pandas as pd
import wfdb
import os
import pickle
from collections import defaultdict
from scipy.io import loadmat
from tqdm import tqdm

DATASET_BASE = '/Users/skandashyam/Desktop/MajorProject/Project/Dataset'

CLASS_NAMES = {
    0: 'Normal',
    1: 'AV Block',
    2: 'Complete Heart Block',
    3: 'RBBB',
    4: 'LBBB'
}

# ============================================================================
# PTB-XL ANALYSIS
# ============================================================================

def analyze_ptbxl():
    """Analyze PTB-XL dataset."""
    print("\n" + "="*80)
    print("PTB-XL ANALYSIS")
    print("="*80)
    
    dataset_path = os.path.join(DATASET_BASE, 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
    csv_path = os.path.join(dataset_path, 'ptbxl_database.csv')
    
    df = pd.read_csv(csv_path)
    
    # SCP Code mapping
    SCP_CODE_MAPPING = {
        '1AVB': 1, '2AVB': 1, '2AVB1': 1, '2AVB2': 1,
        '3AVB': 2,
        'RBBB': 3, 'IRBBB': 3, 'CRBBB': 3,
        'LBBB': 4, 'ILBBB': 4, 'CLBBB': 4,
    }
    
    def extract_scp_label(scp_codes_str):
        try:
            import ast
            codes = ast.literal_eval(scp_codes_str) if isinstance(scp_codes_str, str) else scp_codes_str
            for code in codes.keys():
                if code in SCP_CODE_MAPPING:
                    return SCP_CODE_MAPPING[code]
            return 0
        except:
            return 0
    
    # Extract labels
    df['label'] = df['scp_codes'].apply(extract_scp_label)
    
    # Count patient records per class
    patient_counts = df.groupby('label').size()
    
    print(f"\nTotal Patient Records: {len(df)}")
    print("\nPatient Records per Class:")
    for class_id in range(5):
        count = patient_counts.get(class_id, 0)
        print(f"  {CLASS_NAMES[class_id]}: {count} patients")
    
    return patient_counts

# ============================================================================
# MIT-BIH ANALYSIS
# ============================================================================

def analyze_mitbih():
    """Analyze MIT-BIH dataset."""
    print("\n" + "="*80)
    print("MIT-BIH ANALYSIS")
    print("="*80)
    
    dataset_path = os.path.join(DATASET_BASE, 'mit-bih-arrhythmia-database-1.0.0')
    records_file = os.path.join(dataset_path, 'RECORDS')
    
    if not os.path.exists(records_file):
        print("RECORDS file not found")
        return {}
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    
    print(f"\nTotal Patient Records: {len(record_names)}")
    print("\nPatient Records per Class:")
    print(f"  Normal: {len(record_names)} (all default to Normal)")
    for class_id in range(1, 5):
        print(f"  {CLASS_NAMES[class_id]}: 0")
    
    return {0: len(record_names)}

# ============================================================================
# LUDB ANALYSIS
# ============================================================================

def analyze_ludb():
    """Analyze LUDB dataset."""
    print("\n" + "="*80)
    print("LUDB ANALYSIS")
    print("="*80)
    
    dataset_path = os.path.join(DATASET_BASE, 'lobachevsky-university-electrocardiography-database-1.0.1')
    ludb_csv = os.path.join(dataset_path, 'ludb.csv')
    
    if os.path.exists(ludb_csv):
        df = pd.read_csv(ludb_csv)
        total_records = len(df)
    else:
        total_records = 0
    
    print(f"\nTotal Patient Records: {total_records}")
    print("\nPatient Records per Class:")
    print(f"  Normal: {total_records} (all default to Normal)")
    for class_id in range(1, 5):
        print(f"  {CLASS_NAMES[class_id]}: 0")
    
    return {0: total_records}

# ============================================================================
# PTB DIAGNOSTIC ANALYSIS
# ============================================================================

def analyze_ptb_diagnostic():
    """Analyze PTB Diagnostic dataset."""
    print("\n" + "="*80)
    print("PTB DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    dataset_path = os.path.join(DATASET_BASE, 'ptb-diagnostic-ecg-database-1.0.0')
    
    patient_dirs = sorted([d for d in os.listdir(dataset_path) 
                          if d.startswith('patient') and os.path.isdir(os.path.join(dataset_path, d))])
    
    total_records = len(patient_dirs)
    
    print(f"\nTotal Patient Records: {total_records}")
    print("\nPatient Records per Class:")
    print(f"  Normal: {total_records} (all default to Normal due to complex metadata)")
    for class_id in range(1, 5):
        print(f"  {CLASS_NAMES[class_id]}: 0")
    
    return {0: total_records}

# ============================================================================
# PHYSIONET ANALYSIS
# ============================================================================

def analyze_physionet():
    """Analyze PhysioNet dataset."""
    print("\n" + "="*80)
    print("PHYSIONET ANALYSIS")
    print("="*80)
    
    dataset_path = os.path.join(DATASET_BASE, 'a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0')
    
    SNOMED_TO_CLASS = {
        '270492004': 1, '195042002': 1, '54016002': 1, '28189009': 1,
        '27885002': 2,
        '59118001': 3,
        '164909002': 4,
    }
    
    records_file = os.path.join(dataset_path, 'RECORDS')
    if not os.path.exists(records_file):
        print("RECORDS file not found")
        return {}
    
    with open(records_file, 'r') as f:
        record_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Count by class
    class_counts = defaultdict(int)
    
    for record_path in tqdm(record_names, desc="Processing PhysioNet records"):
        try:
            record_dir = os.path.join(dataset_path, record_path.rstrip('/'))
            hea_files = [f for f in os.listdir(record_dir) if f.endswith('.hea')]
            
            if hea_files:
                hea_file_path = os.path.join(record_dir, hea_files[0])
                label = 0
                
                with open(hea_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#Dx:'):
                            dx_part = line.replace('#Dx:', '').strip()
                            snomed_codes = dx_part.split(',')
                            
                            found = False
                            for snomed_code in snomed_codes:
                                code = snomed_code.strip()
                                if code in SNOMED_TO_CLASS:
                                    if code == '27885002':
                                        label = 2
                                        found = True
                                        break
                                    elif code == '28189009':
                                        label = 1
                                        found = True
                                        break
                                    elif label == 0:
                                        label = SNOMED_TO_CLASS[code]
                                        found = True
                            if found:
                                break
                
                class_counts[label] += 1
        except:
            class_counts[0] += 1
    
    total_records = sum(class_counts.values())
    
    print(f"\nTotal Patient Records: {total_records}")
    print("\nPatient Records per Class:")
    for class_id in range(5):
        count = class_counts.get(class_id, 0)
        print(f"  {CLASS_NAMES[class_id]}: {count}")
    
    return dict(class_counts)

# ============================================================================
# FINAL MERGED DATASET ANALYSIS
# ============================================================================

def analyze_merged():
    """Analyze final merged dataset."""
    print("\n" + "="*80)
    print("FINAL MERGED DATASET ANALYSIS")
    print("="*80)
    
    pkl_path = os.path.join(DATASET_BASE, 'preprocessed_dataset/merged_ecg_dataset_all5_complete.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']
    
    print(f"\nTotal ECG Segments: {len(X):,}")
    print(f"Segment Shape: {X.shape}")
    
    print("\nSegments per Class:")
    for class_id in range(5):
        count = np.sum(y == class_id)
        percentage = (count / len(y)) * 100
        print(f"  {CLASS_NAMES[class_id]}: {count:,} ({percentage:.1f}%)")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run analysis for all datasets."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ECG DATASET ANALYSIS")
    print("="*80)
    
    # Analyze individual datasets
    ptbxl_patients = analyze_ptbxl()
    mitbih_patients = analyze_mitbih()
    ludb_patients = analyze_ludb()
    ptb_diag_patients = analyze_ptb_diagnostic()
    physionet_patients = analyze_physionet()
    
    # Analyze merged dataset
    analyze_merged()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - TOTAL PATIENT RECORDS BY DATASET")
    print("="*80)
    
    datasets_summary = {
        'PTB-XL': int(ptbxl_patients.sum()),
        'MIT-BIH': int(sum(mitbih_patients.values())),
        'LUDB': int(sum(ludb_patients.values())),
        'PTB Diagnostic': int(sum(ptb_diag_patients.values())),
        'PhysioNet': int(sum(physionet_patients.values())),
    }
    
    total_patients = sum(datasets_summary.values())
    
    print(f"\nDataset\t\t\tPatient Records")
    print("-" * 50)
    for name, count in datasets_summary.items():
        print(f"{name:<25}\t{count:,}")
    print("-" * 50)
    print(f"{'TOTAL':<25}\t{total_patients:,}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
