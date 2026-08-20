"""
Advanced ECG analysis: Waveform detection, attention extraction, explainability.
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# WAVEFORM DETECTION
# ============================================================================

class ECGWaveformDetector:
    """Detect QRS complex, P-wave, T-wave in ECG signal."""
    
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
    
    def detect_qrs(self, ecg_signal):
        """Detect QRS complex using Pan-Tompkins algorithm."""
        try:
            # High-pass filter (remove baseline wander)
            sos = signal.butter(4, 5, 'high', fs=self.fs, output='sos')
            filtered = signal.sosfilt(sos, ecg_signal)
            
            # Derivative (to find steep slopes)
            derivative = np.diff(filtered)
            
            # Squaring (emphasize peaks)
            squared = derivative ** 2
            
            # Moving average window
            window_size = int(0.15 * self.fs)  # 150ms window
            moving_avg = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
            
            # Find peaks
            threshold = np.mean(moving_avg) + 1.5 * np.std(moving_avg)
            qrs_peaks, _ = find_peaks(moving_avg, height=threshold, distance=int(0.4*self.fs))
            
            return qrs_peaks.tolist()
        except Exception as e:
            logger.error(f"QRS detection error: {e}")
            return []
    
    def detect_p_wave(self, ecg_signal):
        """Detect P-wave (before QRS)."""
        try:
            qrs_peaks = self.detect_qrs(ecg_signal)
            if not qrs_peaks:
                return []
            
            p_waves = []
            for qrs in qrs_peaks:
                # P-wave is ~100-200ms before QRS
                search_start = max(0, qrs - int(0.2 * self.fs))
                search_end = qrs - int(0.05 * self.fs)
                
                if search_start < search_end:
                    segment = ecg_signal[search_start:search_end]
                    p_peak = search_start + np.argmax(np.abs(segment))
                    p_waves.append(int(p_peak))
            
            return p_waves
        except Exception as e:
            logger.error(f"P-wave detection error: {e}")
            return []
    
    def detect_t_wave(self, ecg_signal):
        """Detect T-wave (after QRS)."""
        try:
            qrs_peaks = self.detect_qrs(ecg_signal)
            if not qrs_peaks:
                return []
            
            t_waves = []
            for qrs in qrs_peaks:
                # T-wave is ~200-400ms after QRS
                search_start = qrs + int(0.2 * self.fs)
                search_end = min(len(ecg_signal), qrs + int(0.4 * self.fs))
                
                if search_start < search_end:
                    segment = ecg_signal[search_start:search_end]
                    t_peak = search_start + np.argmax(np.abs(segment))
                    t_waves.append(int(t_peak))
            
            return t_waves
        except Exception as e:
            logger.error(f"T-wave detection error: {e}")
            return []
    
    def get_all_annotations(self, ecg_signal):
        """Get all waveform annotations."""
        return {
            'qrs': self.detect_qrs(ecg_signal),
            'p_wave': self.detect_p_wave(ecg_signal),
            't_wave': self.detect_t_wave(ecg_signal)
        }


# ============================================================================
# ATTENTION EXTRACTION
# ============================================================================

class AttentionExtractor:
    """Extract attention weights from trained model."""
    
    @staticmethod
    def get_attention_weights(model, signal_tensor):
        """
        Extract attention weights from model's attention layer.
        Works with HybridECGModel that has attention mechanism.
        """
        try:
            import torch
            
            # Forward pass to get attention
            model.eval()
            with torch.no_grad():
                # Get intermediate attention outputs
                # This depends on model architecture
                logits, attention_weights = model(signal_tensor)
                
            if attention_weights is not None:
                # Normalize attention to [0, 1]
                attn = attention_weights.cpu().numpy()
                if attn.ndim > 1:
                    attn = np.mean(attn, axis=0)  # Average across heads
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                return attn
            
            return None
        except Exception as e:
            logger.warning(f"Could not extract attention: {e}")
            return None
    
    @staticmethod
    def get_top_attention_regions(attention_weights, top_k=3):
        """Get regions with highest attention."""
        if attention_weights is None:
            return []
        
        # Find top-k peaks in attention
        peaks, _ = find_peaks(attention_weights, height=np.percentile(attention_weights, 70))
        
        if len(peaks) == 0:
            # Just take top-k indices
            top_indices = np.argsort(attention_weights)[-top_k:]
        else:
            # Take top-k peaks
            peak_values = attention_weights[peaks]
            top_indices = peaks[np.argsort(peak_values)[-top_k:]]
        
        regions = []
        for idx in sorted(top_indices):
            regions.append({
                'position': int(idx),
                'value': float(attention_weights[idx]),
                'percentage': float(100 * idx / len(attention_weights))
            })
        
        return regions


# ============================================================================
# LIME EXPLAINABILITY
# ============================================================================

class ECGExplainer:
    """LIME-like explainability for ECG predictions."""
    
    @staticmethod
    def get_feature_importance(signal, model, signal_length=300, num_samples=100):
        """
        Approximate feature importance using perturbation.
        Show which signal regions most affect the prediction.
        """
        try:
            import torch
            
            original_signal = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).float()
            
            # Get original prediction
            with torch.no_grad():
                logits, _ = model(original_signal)
                original_prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            importance = np.zeros(signal_length)
            window_size = int(signal_length / 10)  # Divide into 10 windows
            
            # Perturb each window and measure change
            for i in range(0, signal_length, window_size):
                end = min(i + window_size, signal_length)
                
                # Create perturbed signal (zero out window)
                perturbed = signal.copy()
                perturbed[i:end] = 0
                
                perturbed_tensor = torch.from_numpy(perturbed).unsqueeze(0).unsqueeze(0).float()
                
                with torch.no_grad():
                    perturbed_logits, _ = model(perturbed_tensor)
                    perturbed_prob = torch.softmax(perturbed_logits, dim=1)[0].cpu().numpy()
                
                # Calculate change in prediction confidence
                prob_change = np.abs(perturbed_prob - original_prob).sum()
                importance[i:end] = prob_change
            
            # Normalize
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            
            return importance.tolist()
        
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return [0] * signal_length


# ============================================================================
# SUMMARY
# ============================================================================

def analyze_ecg_comprehensive(signal, model, sampling_rate=500):
    """
    Comprehensive ECG analysis: waveforms + attention + explainability.
    """
    import torch
    
    result = {}
    
    # 1. Waveform detection
    detector = ECGWaveformDetector(sampling_rate)
    result['waveforms'] = detector.get_all_annotations(signal)
    
    # 2. Attention extraction
    signal_tensor = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).float()
    attention = AttentionExtractor.get_attention_weights(model, signal_tensor)
    result['attention'] = attention.tolist() if attention is not None else None
    result['attention_regions'] = AttentionExtractor.get_top_attention_regions(attention)
    
    # 3. Feature importance (explainability)
    explainer = ECGExplainer()
    result['feature_importance'] = explainer.get_feature_importance(
        signal, model, signal_length=len(signal)
    )
    
    return result
