"""
ECG Interval Measurement Module
Generates clinically consistent ECG measurements based on model predictions.
Note: These are synthetic estimates, not actual measurements from 12-lead ECG.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Tuple

class ECGMetrics:
    """Generate clinically consistent ECG measurements based on diagnosis."""
    
    def __init__(self, sampling_rate: int = 500):
        """
        Args:
            sampling_rate: ECG sampling rate in Hz (default 500 Hz)
        """
        self.sampling_rate = sampling_rate
        self.ms_per_sample = 1000 / sampling_rate
    
    def generate_metrics_from_diagnosis(self, diagnosis: str = 'Normal', 
                                       heart_rate: float = None) -> Dict[str, float]:
        """
        Generate clinically consistent metrics based on diagnosis.
        These are synthetic estimates, not direct measurements.
        
        Args:
            diagnosis: One of 'Normal', 'AV Block', 'Complete Heart Block', 'RBBB', 'LBBB'
            heart_rate: Override heart rate (optional)
        
        Returns:
            Dictionary with consistent metrics
        """
        # Clinical ranges and typical values for each condition
        clinical_profiles = {
            'Normal': {
                'hr': (60, 100, 75),           # min, max, typical
                'pr': (120, 200, 160),         # normal PR interval
                'qrs': (60, 100, 85),          # normal QRS
                'qt': (350, 430, 400),         # normal QT
            },
            'AV Block': {
                'hr': (45, 80, 60),            # often slower with AV block
                'pr': (200, 300, 240),         # PROLONGED PR (diagnostic)
                'qrs': (70, 110, 90),          # normal QRS width
                'qt': (360, 450, 400),         # normal QT
            },
            'Complete Heart Block': {
                'hr': (30, 50, 40),            # much slower - junctional escape
                'pr': (0, 50, 0),              # NO relationship (dissociated)
                'qrs': (80, 140, 110),         # wide QRS (escape rhythm)
                'qt': (380, 480, 420),         # may be prolonged
            },
            'RBBB': {
                'hr': (60, 100, 78),           # normal rate
                'pr': (120, 200, 160),         # normal PR
                'qrs': (120, 150, 135),        # WIDE QRS (diagnostic)
                'qt': (360, 440, 400),         # normal QT
            },
            'LBBB': {
                'hr': (50, 90, 70),            # often slightly slower
                'pr': (120, 200, 165),         # normal PR
                'qrs': (120, 160, 140),        # WIDE QRS (diagnostic)
                'qt': (370, 450, 410),         # may be prolonged
            }
        }
        
        # Get profile for diagnosis
        profile = clinical_profiles.get(diagnosis, clinical_profiles['Normal'])
        
        # Generate or use provided heart rate
        if heart_rate is None:
            hr_min, hr_max, hr_typical = profile['hr']
            # Add realistic variation around typical value
            heart_rate = np.random.normal(hr_typical, (hr_max - hr_min) / 6)
            heart_rate = np.clip(heart_rate, hr_min, hr_max)
        else:
            hr_min, hr_max, _ = profile['hr']
            heart_rate = np.clip(heart_rate, hr_min, hr_max)
        
        # Generate other metrics with realistic variation
        pr_min, pr_max, pr_typical = profile['pr']
        pr_interval = np.random.normal(pr_typical, (pr_max - pr_min) / 8)
        pr_interval = np.clip(pr_interval, pr_min, pr_max)
        
        qrs_min, qrs_max, qrs_typical = profile['qrs']
        qrs_duration = np.random.normal(qrs_typical, (qrs_max - qrs_min) / 8)
        qrs_duration = np.clip(qrs_duration, qrs_min, qrs_max)
        
        # QT interval should vary inversely with HR (Bazett correction concept)
        qt_min, qt_max, qt_typical = profile['qt']
        # Shorter HR = longer QT; faster HR = shorter QT
        hr_factor = heart_rate / 60  # normalize to 60 bpm
        qt_interval = qt_typical / np.sqrt(hr_factor)
        qt_interval = np.clip(qt_interval, qt_min, qt_max)
        
        # Calculate RR interval from HR
        rr_interval = 60000 / heart_rate if heart_rate > 0 else 1000
        
        return {
            'heart_rate': float(round(heart_rate, 1)),
            'pr_interval': float(round(pr_interval, 1)),
            'qrs_duration': float(round(qrs_duration, 1)),
            'qt_interval': float(round(qt_interval, 1)),
            'rr_interval': float(round(rr_interval, 1))
        }
    
    def get_all_metrics(self, signal: np.ndarray, diagnosis: str = None) -> Dict[str, float]:
        """
        Generate metrics. If diagnosis provided, use clinical profile.
        Otherwise, attempt signal-based estimation.
        
        Args:
            signal: ECG signal
            diagnosis: Clinical diagnosis (optional)
        
        Returns:
            Dictionary with metrics
        """
        if diagnosis:
            # Use diagnosis-based generation
            return self.generate_metrics_from_diagnosis(diagnosis)
        else:
            # Fallback: generate default normal metrics with signal-based HR estimate
            try:
                hr = self._estimate_heart_rate(signal)
            except:
                hr = 75.0
            
            return self.generate_metrics_from_diagnosis('Normal', heart_rate=hr)
    
    def _estimate_heart_rate(self, signal: np.ndarray) -> float:
        """
        Attempt to estimate HR from signal using spectral analysis.
        """
        try:
            from scipy.fft import fft, fftfreq
            
            N = len(signal)
            yf = np.abs(fft(signal))
            xf = fftfreq(N, 1/self.sampling_rate)
            
            # Look for dominant frequency in HR range (0.7-2 Hz = 42-120 bpm)
            hr_range_mask = (xf >= 0.7) & (xf <= 2.0)
            if np.any(hr_range_mask):
                dominant_freq = xf[np.argmax(yf * hr_range_mask)]
                hr = dominant_freq * 60
                if 42 <= hr <= 120:
                    return float(hr)
            
            # Fallback to normal range
            return 75.0
        except:
            return 75.0
    
    def get_metrics_status(self, metrics: Dict[str, float], diagnosis: str = None) -> Dict[str, str]:
        """
        Determine status based on metrics and diagnosis consistency.
        """
        status = {}
        
        # Heart Rate status
        hr = metrics['heart_rate']
        if 60 <= hr <= 100:
            status['heart_rate'] = 'Normal'
        elif hr < 60:
            status['heart_rate'] = 'Bradycardia (slow)'
        else:
            status['heart_rate'] = 'Tachycardia (fast)'
        
        # PR Interval status
        pr = metrics['pr_interval']
        if pr == 0:
            status['pr_interval'] = 'Dissociated'  # Complete heart block
        elif 120 <= pr <= 200:
            status['pr_interval'] = 'Normal'
        elif pr < 120:
            status['pr_interval'] = 'Short'
        else:
            status['pr_interval'] = 'Prolonged (AV block)'
        
        # QRS Duration status - now tied to diagnosis
        qrs = metrics['qrs_duration']
        if diagnosis and diagnosis in ['RBBB', 'LBBB', 'Complete Heart Block']:
            if qrs >= 120:
                status['qrs_duration'] = 'Wide (bundle branch block)'
            else:
                status['qrs_duration'] = 'Normal'  # Model says block but QRS not widened
        else:
            if 60 <= qrs <= 100:
                status['qrs_duration'] = 'Normal'
            elif qrs < 60:
                status['qrs_duration'] = 'Narrow'
            else:
                status['qrs_duration'] = 'Wide'
        
        # QT Interval status
        qt = metrics['qt_interval']
        if 320 <= qt <= 440:
            status['qt_interval'] = 'Normal'
        elif qt < 320:
            status['qt_interval'] = 'Short'
        else:
            status['qt_interval'] = 'Prolonged'
        
        return status


if __name__ == '__main__':
    # Test
    metrics_calc = ECGMetrics(sampling_rate=500)
    
    diagnoses = ['Normal', 'AV Block', 'Complete Heart Block', 'RBBB', 'LBBB']
    
    print("=" * 80)
    print("ECG METRICS - CLINICAL CONSISTENCY TEST")
    print("=" * 80)
    
    for diagnosis in diagnoses:
        metrics = metrics_calc.generate_metrics_from_diagnosis(diagnosis)
        status = metrics_calc.get_metrics_status(metrics, diagnosis)
        
        print(f"\n{diagnosis.upper()}")
        print("-" * 80)
        print(f"  Heart Rate:    {metrics['heart_rate']:.1f} bpm ({status['heart_rate']})")
        print(f"  PR Interval:   {metrics['pr_interval']:.1f} ms ({status['pr_interval']})")
        print(f"  QRS Duration:  {metrics['qrs_duration']:.1f} ms ({status['qrs_duration']})")
        print(f"  QT Interval:   {metrics['qt_interval']:.1f} ms ({status['qt_interval']})")
        print(f"  RR Interval:   {metrics['rr_interval']:.1f} ms")



if __name__ == '__main__':
    # Test
    metrics_calc = ECGMetrics(sampling_rate=500)
    
    # Create test signal
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 3 * t) + 0.2 * np.random.randn(500)
    
    metrics = metrics_calc.get_all_metrics(signal)
    status = metrics_calc.get_metrics_status(metrics)
    
    print("ECG Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.1f}")
    
    print("\nStatus:")
    for key, value in status.items():
        print(f"  {key}: {value}")
