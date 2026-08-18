"""
Flask backend for ECG Binary and Multi-class Classification Pipeline.
Minimal, production-ready inference server.

Supports: .npz, .npy, .csv, .txt, .dat, .xlsx, .mat
"""

import os
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template, make_response
from werkzeug.utils import secure_filename
import json

# Import inference engine
from inference_engine import create_engine
from ecg_metrics import ECGMetrics

# ============================================================================
# SETUP
# ============================================================================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = Path('uploads').mkdir(exist_ok=True) or 'uploads'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
try:
    engine = create_engine(device='cpu')
    logger.info("✓ Inference engine initialized")
except Exception as e:
    logger.error(f"Failed to initialize inference engine: {e}")
    engine = None

# Initialize metrics calculator
metrics_calc = ECGMetrics(sampling_rate=500)

ALLOWED_EXTENSIONS = {'.npz', '.npy', '.csv', '.txt', '.dat', '.xlsx', '.mat'}


# ============================================================================
# CORS HELPER
# ============================================================================

def _add_cors(response):
    """Add CORS headers for local development."""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


@app.after_request
def after_request(response):
    return _add_cors(response)


@app.route('/', methods=['OPTIONS'])
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path=''):
    """Handle CORS preflight requests."""
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response, 200


# ============================================================================
# SIGNAL LOADING HELPER
# ============================================================================

def _load_signal_from_file(filepath: Path) -> np.ndarray:
    """
    Load ECG signal from file. Supports all standard formats.

    Args:
        filepath: Path to the uploaded file

    Returns:
        1D or 2D numpy array of signal data

    Raises:
        ValueError: If format is unsupported or file cannot be read
    """
    suffix = filepath.suffix.lower()

    if suffix == '.npz':
        data = np.load(str(filepath))
        # Try common keys
        for key in ['X', 'signal', 'ecg', 'data']:
            if key in data:
                return data[key]
        # Fall back to first array
        return list(data.values())[0]

    elif suffix == '.npy':
        return np.load(str(filepath))

    elif suffix in ('.csv', '.txt', '.dat'):
        try:
            return np.loadtxt(str(filepath), delimiter=',')
        except Exception:
            return np.loadtxt(str(filepath), delimiter=None)

    elif suffix == '.xlsx':
        import pandas as pd
        df = pd.read_excel(str(filepath))
        return df.iloc[:, 0].values

    elif suffix == '.mat':
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(filepath))
            # Try common ECG variable names from MATLAB files
            for key in ['val', 'ecg', 'signal', 'ECG', 'data', 'x']:
                if key in mat and isinstance(mat[key], np.ndarray):
                    return mat[key]
            # Fall back: take first non-metadata key
            keys = [k for k in mat.keys() if not k.startswith('_')]
            if keys:
                return mat[keys[0]]
            raise ValueError("No valid signal array found in .mat file")
        except ImportError:
            raise ValueError("scipy is required for .mat file support. Install with: pip install scipy")

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: {', '.join(ALLOWED_EXTENSIONS)}")


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render main UI page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Accepts:
    - ECG signal as NPZ/NPY/CSV/DAT/TXT/XLSX/MAT file (multipart form)
    - Or raw signal as JSON: {"signal": [...]}

    Returns:
    - Binary classification result
    - Multi-class classification result
    - ECG metrics
    - Recommendation
    - Signal preview (first 300 samples, downsampled if needed)
    """
    try:
        if engine is None:
            return jsonify({'error': 'Inference engine not initialized. Check server logs.'}), 500

        signal = None

        # ---- File Upload ----
        if 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            filename = secure_filename(file.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                return jsonify({'error': f'Unsupported format: {suffix}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))

            try:
                signal = _load_signal_from_file(filepath)
            except ValueError as ve:
                return jsonify({'error': str(ve)}), 400
            finally:
                try:
                    filepath.unlink(missing_ok=True)
                except Exception:
                    pass

        # ---- JSON Raw Signal ----
        elif request.json and 'signal' in request.json:
            signal = np.array(request.json['signal'], dtype=np.float32)

        else:
            return jsonify({'error': 'No signal provided. Upload a file or send {"signal": [...]}'}), 400

        # Ensure signal is 1D (take first row if 2D batch)
        if signal is not None and signal.ndim > 1:
            signal = signal.flatten() if signal.shape[0] == 1 else signal[0]

        signal = signal.astype(np.float32)
        logger.info(f"Signal shape: {signal.shape}, dtype: {signal.dtype}")

        # ---- Inference ----
        report = engine.predict(signal)
        logger.info(f"✓ Inference complete. Keys: {list(report.keys())}")

        # ---- Timestamp ----
        report['timestamp'] = datetime.now().isoformat()

        # ---- ECG Metrics ----
        # Get the diagnosis from multi-class prediction
        multiclass_diagnosis = report.get('multiclass_classification', {}).get('class_name', 'Normal')
        logger.info(f"Diagnosis for metrics: {multiclass_diagnosis}")
        
        try:
            metrics = metrics_calc.get_all_metrics(signal, diagnosis=multiclass_diagnosis)
            metrics_status = metrics_calc.get_metrics_status(metrics, diagnosis=multiclass_diagnosis)
            report['ecg_metrics'] = metrics
            report['metrics_status'] = metrics_status
            logger.info(f"✓ ECG metrics added: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Could not calculate ECG metrics: {e}", exc_info=True)
            report['ecg_metrics'] = {}
            report['metrics_status'] = {}

        # ---- Signal Preview (for canvas rendering) ----
        # Return up to 300 points for the frontend canvas
        try:
            preview_signal = signal[:300] if len(signal) >= 300 else signal
            # Normalize to [-1, 1] for display
            sig_min, sig_max = float(preview_signal.min()), float(preview_signal.max())
            sig_range = sig_max - sig_min
            if sig_range > 1e-8:
                preview_normalized = ((preview_signal - sig_min) / sig_range * 2 - 1).tolist()
            else:
                preview_normalized = preview_signal.tolist()
            report['signal_preview'] = preview_normalized
            logger.info(f"✓ Signal preview added: {len(report['signal_preview'])} samples")
        except Exception as e:
            logger.error(f"Could not generate signal preview: {e}", exc_info=True)
            report['signal_preview'] = []

        logger.info(f"Final report keys: {list(report.keys())}")
        return jsonify(report), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview', methods=['POST'])
def preview():
    """
    Lightweight signal preview endpoint.
    Returns only the normalized signal array for canvas rendering.
    No model inference is performed.
    """
    try:
        signal = None

        if 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))

            try:
                signal = _load_signal_from_file(filepath)
            except ValueError as ve:
                return jsonify({'error': str(ve)}), 400
            finally:
                try:
                    filepath.unlink(missing_ok=True)
                except Exception:
                    pass

        elif request.json and 'signal' in request.json:
            signal = np.array(request.json['signal'], dtype=np.float32)
        else:
            return jsonify({'error': 'No signal provided'}), 400

        if signal is not None and signal.ndim > 1:
            signal = signal.flatten() if signal.shape[0] == 1 else signal[0]

        signal = signal.astype(np.float32)

        # Downsample to 300 points for display
        if len(signal) > 300:
            indices = np.linspace(0, len(signal) - 1, 300, dtype=int)
            signal = signal[indices]

        sig_min, sig_max = float(signal.min()), float(signal.max())
        sig_range = sig_max - sig_min
        if sig_range > 1e-8:
            normalized = ((signal - sig_min) / sig_range * 2 - 1).tolist()
        else:
            normalized = signal.tolist()

        return jsonify({
            'signal': normalized,
            'length': len(normalized),
            'original_min': sig_min,
            'original_max': sig_max
        }), 200

    except Exception as e:
        logger.error(f"Preview error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok' if engine else 'error',
        'binary_model': engine.binary_predictor.model is not None if engine else False,
        'multiclass_model': engine.multiclass_predictor.model is not None if engine else False,
        'message': 'Engine initialized successfully' if engine else 'Engine failed to initialize'
    }), 200


@app.route('/api/info', methods=['GET'])
def info():
    """Get system information."""
    return jsonify({
        'title': 'ECG Cardiac Block Detection',
        'version': '2.0.0',
        'binary_classes': ['Normal', 'Abnormal'],
        'multiclass_classes': [
            'Normal',
            'AV Block',
            'Complete Heart Block',
            'RBBB',
            'LBBB'
        ],
        'signal_specs': {
            'length': 300,
            'sampling_rate': 500,
            'units': 'mV'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'engine_status': 'Ready' if engine else 'Failed'
    }), 200


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_recommendation(binary_result: dict, multiclass_result: dict) -> dict:
    """Generate clinical recommendations based on predictions."""

    recommendations = {
        'Normal': {
            'status': 'Normal',
            'action': 'No action required',
            'follow_up': 'Routine monitoring'
        },
        'AV Block': {
            'status': 'Abnormal',
            'action': 'Consult cardiologist',
            'follow_up': 'Follow-up ECG in 1–2 weeks'
        },
        'Complete Heart Block': {
            'status': 'Critical',
            'action': 'Urgent cardiology consultation',
            'follow_up': 'Immediate evaluation required'
        },
        'RBBB': {
            'status': 'Abnormal',
            'action': 'Consult cardiologist',
            'follow_up': 'Follow-up ECG as recommended'
        },
        'LBBB': {
            'status': 'Abnormal',
            'action': 'Consult cardiologist',
            'follow_up': 'Follow-up ECG as recommended'
        }
    }

    multiclass_name = multiclass_result.get('class_name', 'Unknown')
    return recommendations.get(multiclass_name, recommendations['Normal'])


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large. Maximum 50MB allowed.'}), 413


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("═" * 60)
    logger.info("CardioScan ECG Classification Backend v2.0")
    logger.info("═" * 60)
    if engine:
        logger.info("✓ Models loaded successfully")
        logger.info("✓ Ready to accept predictions")
    else:
        logger.warning("⚠ Warning: Some models failed to load")
        logger.warning("  Inference may not work correctly")
    logger.info("═" * 60)
    
    # Railway uses PORT environment variable, fallback to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0' if port != 5000 else '127.0.0.1'
    
    logger.info(f"Server: http://{host}:{port}")
    logger.info("Supported formats: " + ", ".join(ALLOWED_EXTENSIONS))
    logger.info("═" * 60)

    app.run(debug=False, host=host, port=port)
