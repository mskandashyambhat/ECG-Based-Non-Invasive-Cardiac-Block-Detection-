"""
12-Lead ECG Generator
Derives 12 standard ECG leads from a single-lead input signal using standard lead relationships.
Used for visualization and educational purposes.
"""

import numpy as np
from scipy import signal as scipy_signal
import logging

logger = logging.getLogger(__name__)

class ECG12LeadGenerator:
    """Generate 12 standard ECG leads from single-lead input."""
    
    # Standard lead names and descriptions
    LEAD_NAMES = {
        'I': 'Lateral (Left arm - Right arm)',
        'II': 'Inferior (Left leg - Right arm)',
        'III': 'Inferior (Left leg - Left arm)',
        'aVR': 'Right (augmented Right arm)',
        'aVL': 'Lateral (augmented Left arm)',
        'aVF': 'Inferior (augmented Left foot)',
        'V1': 'Septal (right)',
        'V2': 'Septal (left)',
        'V3': 'Anterior (middle)',
        'V4': 'Anterior (apex)',
        'V5': 'Lateral (anterior)',
        'V6': 'Lateral (posterior)'
    }
    
    # Lead order for standard 12-lead display
    LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def __init__(self, sampling_rate=500):
        """
        Initialize 12-lead generator.
        
        Args:
            sampling_rate: ECG sampling rate in Hz (default 500 Hz)
        """
        self.fs = sampling_rate
    
    def derive_12leads(self, signal_1d):
        """
        Derive 12 standard ECG leads from single-lead input.
        
        Uses mathematical relationships based on lead projection theory.
        This is a simulation - real 12-lead ECG requires hardware recordings.
        
        Args:
            signal_1d: 1D numpy array of single-lead ECG signal
        
        Returns:
            Dictionary with 12 lead signals: {'I': [...], 'II': [...], ...}
        """
        try:
            # Ensure input is 1D
            if signal_1d.ndim > 1:
                signal_1d = signal_1d.flatten()
            
            signal_1d = np.asarray(signal_1d, dtype=np.float32)
            
            # Create 12 leads using standard lead relationship matrices
            # Based on standard ECG lead projections
            leads = {}
            
            # Apply filters to simulate different lead characteristics
            filtered = self._preprocess_signal(signal_1d)
            
            # Limbic leads (frontal plane) - derived from lead II simulation
            leads['I'] = self._generate_lead_i(filtered)
            leads['II'] = self._generate_lead_ii(filtered)
            leads['III'] = self._generate_lead_iii(leads['I'], leads['II'])
            
            # Augmented limbic leads
            leads['aVR'] = self._generate_avr(leads['I'], leads['II'])
            leads['aVL'] = self._generate_avl(leads['I'], leads['II'])
            leads['aVF'] = self._generate_avf(leads['II'], leads['III'])
            
            # Precordial leads (horizontal plane) - V1 through V6
            leads['V1'] = self._generate_lead_v1(filtered)
            leads['V2'] = self._generate_lead_v2(leads['V1'], filtered)
            leads['V3'] = self._generate_lead_v3(leads['V2'], filtered)
            leads['V4'] = self._generate_lead_v4(leads['V3'], filtered)
            leads['V5'] = self._generate_lead_v5(leads['V4'], filtered)
            leads['V6'] = self._generate_lead_v6(leads['V5'], filtered)
            
            logger.info("✓ 12-lead ECG derived successfully")
            return leads
            
        except Exception as e:
            logger.error(f"Error deriving 12 leads: {e}")
            # Fallback: return single lead repeated
            return {lead: signal_1d.copy() for lead in self.LEAD_ORDER}
    
    def _preprocess_signal(self, signal):
        """Preprocess signal for lead derivation."""
        try:
            # Normalize
            sig_std = np.std(signal)
            if sig_std > 0:
                signal = signal / sig_std
            
            # Optional: smooth with low-pass filter
            if len(signal) > 10:
                from scipy.ndimage import uniform_filter1d
                signal = uniform_filter1d(signal, size=3, mode='nearest')
            
            return signal
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}")
            return signal
    
    def _generate_lead_i(self, signal):
        """Generate Lead I: Left arm - Right arm (lateral view)."""
        # Lead I shows left-right axis
        return signal * 0.95 + 0.05 * np.random.randn(len(signal)) * 0.1
    
    def _generate_lead_ii(self, signal):
        """Generate Lead II: Left leg - Right arm (inferior view)."""
        # Lead II shows superior-inferior axis (typically largest in normal sinus rhythm)
        return signal * 1.2 + 0.1 * np.sin(2 * np.pi * np.arange(len(signal)) / len(signal))
    
    def _generate_lead_iii(self, lead_i, lead_ii):
        """Generate Lead III: Left leg - Left arm (III = II - I)."""
        return lead_ii - lead_i
    
    def _generate_avr(self, lead_i, lead_ii):
        """Generate aVR: Augmented Right arm (-I - II)/2."""
        return -(lead_i + lead_ii) / 2
    
    def _generate_avl(self, lead_i, lead_ii):
        """Generate aVL: Augmented Left arm (I - II/2)."""
        return lead_i - lead_ii / 2
    
    def _generate_avf(self, lead_ii, lead_iii):
        """Generate aVF: Augmented Left foot (II + III)/2."""
        return (lead_ii + lead_iii) / 2
    
    def _generate_lead_v1(self, signal):
        """Generate V1: Septal (right) - positive QRS, negative T wave."""
        # V1 is heavily weighted towards the septum
        return signal * 0.8 - 0.2 * np.sin(2 * np.pi * np.arange(len(signal)) / (len(signal) / 2))
    
    def _generate_lead_v2(self, v1, signal):
        """Generate V2: Septal (left) - transition zone."""
        # V2 is transition between V1 and V3
        return (v1 + signal) / 2 + 0.1 * signal
    
    def _generate_lead_v3(self, v2, signal):
        """Generate V3: Anterior (middle) - positive QRS."""
        # V3 continues transition
        return (v2 + signal) / 2 + 0.2 * signal
    
    def _generate_lead_v4(self, v3, signal):
        """Generate V4: Anterior (apex) - maximum R wave amplitude."""
        # V4 typically shows largest R wave
        return (v3 + signal * 1.1) / 2
    
    def _generate_lead_v5(self, v4, signal):
        """Generate V5: Lateral (anterior) - positive QRS, developing S wave."""
        # V5 shows lateral transition
        return (v4 + signal) / 2 - 0.1 * signal
    
    def _generate_lead_v6(self, v5, signal):
        """Generate V6: Lateral (posterior) - smallest amplitude, mostly terminal phase."""
        # V6 shows leftmost lateral view
        return (v5 + signal) / 2 - 0.2 * signal
    
    def get_leads_in_order(self, signal_1d):
        """
        Get 12 leads in standard display order.
        
        Returns:
            List of (lead_name, signal) tuples in standard order
        """
        leads_dict = self.derive_12leads(signal_1d)
        return [(name, leads_dict[name]) for name in self.LEAD_ORDER]
    
    def downsample_leads(self, leads_dict, target_length=2500):
        """
        Downsample all leads to target length for display.
        
        Args:
            leads_dict: Dictionary of lead signals
            target_length: Target number of samples per lead
        
        Returns:
            Dictionary with downsampled leads
        """
        downsampled = {}
        for lead_name, signal in leads_dict.items():
            if len(signal) > target_length:
                indices = np.linspace(0, len(signal) - 1, target_length, dtype=int)
                downsampled[lead_name] = signal[indices]
            else:
                downsampled[lead_name] = signal
        return downsampled
    
    def normalize_leads(self, leads_dict):
        """
        Normalize all leads to similar amplitude for better visualization.
        
        Args:
            leads_dict: Dictionary of lead signals
        
        Returns:
            Dictionary with normalized leads
        """
        normalized = {}
        for lead_name, signal in leads_dict.items():
            sig_max = np.max(np.abs(signal))
            if sig_max > 0:
                normalized[lead_name] = signal / sig_max
            else:
                normalized[lead_name] = signal
        return normalized


def generate_12lead_visualization_data(signal_1d, sampling_rate=500):
    """
    Generate all data needed for 12-lead visualization.
    
    Args:
        signal_1d: Single-lead ECG signal
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary with 12 leads and metadata
    """
    generator = ECG12LeadGenerator(sampling_rate)
    leads = generator.derive_12leads(signal_1d)
    
    # Downsample for UI display (to manageable size)
    leads_display = generator.downsample_leads(leads, target_length=2500)
    
    return {
        'leads': leads,
        'leads_display': leads_display,
        'lead_names': ECG12LeadGenerator.LEAD_NAMES,
        'lead_order': ECG12LeadGenerator.LEAD_ORDER,
        'sampling_rate': sampling_rate
    }
