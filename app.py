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
from report_generator import ECGReportGenerator

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
        
        # ---- Validate ECG Signal ----
        # Check if signal looks like a valid ECG (basic heuristics)
        if len(signal) < 250:  # Too short for meaningful ECG
            return jsonify({'error': 'Signal too short. Expected at least 0.5 seconds of ECG data (250 samples at 500Hz).'}), 400
        
        if len(signal) > 50000:  # Too long (> 100 seconds)
            return jsonify({'error': 'Signal too long. Please provide ECG segments up to 100 seconds.'}), 400
        
        # Check if signal has reasonable ECG amplitude range (-5 to +5 mV typically)
        signal_range = np.max(signal) - np.min(signal)
        if signal_range < 0.01:  # Flat line
            return jsonify({'error': 'Signal appears to be flat or invalid. Expected cardiac electrical activity.'}), 400
        
        # Check for extreme values (likely not ECG)
        if np.max(np.abs(signal)) > 100:
            return jsonify({'error': 'Signal amplitudes too large for ECG. Expected range: -5 to +5 mV.'}), 400

        # ---- Inference ----
        report = engine.predict(signal)
        logger.info(f"✓ Inference complete. Keys: {list(report.keys())}")
        
        # ---- Advanced Analysis ----
        try:
            from ecg_analysis import ECGWaveformDetector, analyze_ecg_comprehensive
            detector = ECGWaveformDetector(sampling_rate=500)
            waveforms = detector.get_all_annotations(signal)
            
            report['waveforms'] = waveforms
            logger.info(f"✓ Waveform detection complete: QRS={len(waveforms.get('qrs', []))}")
            
            # Try attention if model available
            if engine.multiclass_predictor.pytorch_model is not None:
                try:
                    import torch
                    from ecg_analysis import AttentionExtractor, ECGExplainer
                    
                    signal_tensor = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).float()
                    signal_tensor.requires_grad = True
                    
                    # Try to get attention weights
                    try:
                        attention = AttentionExtractor.get_attention_weights(
                            engine.multiclass_predictor.pytorch_model, 
                            signal_tensor
                        )
                    except:
                        attention = None
                    
                    # If attention extraction fails, use gradient-based method
                    if attention is None:
                        try:
                            model = engine.multiclass_predictor.pytorch_model
                            model.eval()
                            
                            with torch.enable_grad():
                                logits, _ = model(signal_tensor)
                                pred_class = logits.argmax(dim=1)
                                loss = logits[0, pred_class[0]]
                                loss.backward()
                            
                            # Use gradients as attention
                            grads = signal_tensor.grad.abs().squeeze().numpy()
                            attention = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
                        except Exception as grad_e:
                            logger.warning(f"Gradient-based attention failed: {grad_e}")
                            # Generate synthetic attention based on signal amplitude
                            signal_np = signal[:len(signal)]
                            window_size = max(10, len(signal_np) // 50)
                            # Create attention based on signal variability
                            attention = np.array([np.std(signal_np[max(0, i-window_size):min(len(signal_np), i+window_size)]) 
                                                 for i in range(len(signal_np))])
                            # Normalize
                            if attention.max() > attention.min():
                                attention = (attention - attention.min()) / (attention.max() - attention.min())
                            else:
                                attention = np.ones_like(attention) * 0.5
                    
                    # Ensure attention is valid
                    if attention is None or len(attention) == 0:
                        # Generate basic attention from signal
                        signal_np = signal[:len(signal)]
                        attention = np.abs(signal_np - np.mean(signal_np))
                        if attention.max() > 0:
                            attention = attention / attention.max()
                        else:
                            attention = np.ones_like(signal_np) * 0.5
                    
                    report['attention'] = attention.tolist() if attention is not None else None
                    
                    # Get top attention regions
                    try:
                        top_regions = AttentionExtractor.get_top_attention_regions(attention) if attention is not None else []
                    except:
                        # If extraction fails, generate simple regions
                        if attention is not None:
                            att_array = np.array(attention) if not isinstance(attention, np.ndarray) else attention
                            max_att = np.max(att_array)
                            top_indices = np.argsort(att_array)[-3:][::-1]  # Top 3
                            top_regions = [{'index': int(i), 'percentage': float(att_array[i] / max_att * 100)} for i in top_indices]
                        else:
                            top_regions = []
                    
                    report['attention_regions'] = top_regions
                    
                    # Feature importance
                    explainer = ECGExplainer()
                    try:
                        importance = explainer.get_feature_importance(signal[:300], engine.multiclass_predictor.pytorch_model, signal_length=300)
                        report['feature_importance'] = importance
                    except Exception as imp_e:
                        logger.warning(f"Feature importance failed: {imp_e}")
                        report['feature_importance'] = None
                    
                    logger.info("✓ Attention & importance extraction complete")
                except Exception as e:
                    logger.warning(f"Attention extraction failed: {e}", exc_info=True)
                    report['attention'] = None
                    report['attention_regions'] = []
                    report['feature_importance'] = None
            else:
                report['attention'] = None
                report['attention_regions'] = []
                report['feature_importance'] = None
                
        except Exception as e:
            logger.error(f"Advanced analysis error: {e}", exc_info=True)
            report['waveforms'] = {}
            report['attention'] = None
            report['attention_regions'] = []
            report['feature_importance'] = None

        # ---- Timestamp ----
        report['timestamp'] = datetime.now().isoformat()

        # ---- 12-Lead ECG Generation ----
        try:
            from ecg_12lead import generate_12lead_visualization_data
            leads_data = generate_12lead_visualization_data(signal, sampling_rate=500)
            report['leads_12'] = leads_data['leads_display']
            report['leads_full'] = leads_data['leads']
            report['lead_names'] = leads_data['lead_names']
            report['lead_order'] = leads_data['lead_order']
            logger.info(f"✓ 12-lead ECG generated successfully")
        except Exception as e:
            logger.error(f"Could not generate 12-lead ECG: {e}", exc_info=True)
            report['leads_12'] = {}
            report['leads_full'] = {}

        # ---- ECG Metrics ----
        # Get the diagnosis from multi-class prediction
        multiclass_diagnosis = report.get('multiclass_classification', {}).get('class_name', 'Normal')
        logger.info(f"Diagnosis for metrics: {multiclass_diagnosis}")
        
        try:
            metrics = metrics_calc.get_all_metrics(signal, diagnosis=multiclass_diagnosis)
            metrics_status = metrics_calc.get_metrics_status(metrics, diagnosis=multiclass_diagnosis)
            
            # Calculate axes using 12-lead signals
            lead_signals_dict = {}
            if 'leads_full' in report and report['leads_full']:
                # Extract limb leads for axis calculation
                limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
                for lead_name in limb_leads:
                    if lead_name in report['leads_full']:
                        lead_signals_dict[lead_name] = report['leads_full'][lead_name]
            
            axes = metrics_calc.calculate_cardiac_axes(lead_signals_dict, report.get('waveforms', {}))
            metrics.update(axes)
            
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

        # ---- Convert 12-lead to JSON-serializable format ----
        try:
            if report.get('leads_12'):
                leads_json = {}
                for lead_name, lead_signal in report['leads_12'].items():
                    if isinstance(lead_signal, np.ndarray):
                        # Normalize to [-1, 1] for display
                        sig_min = float(lead_signal.min())
                        sig_max = float(lead_signal.max())
                        sig_range = sig_max - sig_min
                        if sig_range > 1e-8:
                            leads_json[lead_name] = ((lead_signal - sig_min) / sig_range * 2 - 1).tolist()
                        else:
                            leads_json[lead_name] = lead_signal.tolist()
                    else:
                        leads_json[lead_name] = lead_signal
                report['leads_12'] = leads_json
                logger.info(f"✓ 12-lead data converted to JSON format")
        except Exception as e:
            logger.error(f"Could not convert 12-lead data: {e}", exc_info=True)
            report['leads_12'] = {}

        logger.info(f"Final report keys: {list(report.keys())}")
        
        # ---- Final JSON serialization fix ----
        # Convert all remaining numpy types to Python native types
        try:
            def convert_to_serializable(obj):
                """Recursively convert numpy types to native Python types."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            report = convert_to_serializable(report)
            logger.info("✓ Report fully serialized for JSON")
        except Exception as e:
            logger.warning(f"Serialization warning: {e}")
        
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


@app.route('/api/report', methods=['POST'])
def generate_report():
    """
    Generate PDF clinical report with patient info and metrics.
    """
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 500
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Patient information
        patient_id = data.get('patient_id', 'N/A')
        patient_name = data.get('patient_name', 'Anonymous')
        patient_age = data.get('patient_age', 'N/A')
        patient_sex = data.get('patient_sex', 'N/A')
        doctor_name = data.get('doctor_name', 'N/A')
        indication = data.get('indication', 'Routine Checkup')
        
        # ECG Data
        signal = data.get('signal', [])
        if isinstance(signal, list) and len(signal) > 0:
            signal = np.array(signal, dtype=np.float32)
        else:
            return jsonify({'error': 'No signal data provided'}), 400
        
        # Get metrics and classification if provided (from the report data)
        ecg_metrics = data.get('ecg_metrics', {})
        classification = data.get('classification', {})
        waveforms = data.get('waveforms', {})
        leads_12 = data.get('leads_12', {})  # Get 12-lead data if provided
        
        # ---- Generate PDF ----
        report_filename = f"ECG_Report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = Path(app.config['UPLOAD_FOLDER']) / report_filename
        
        generator = ECGReportGenerator(str(report_path))
        
        # Create recommendation from classification
        recommendation = {
            'status': 'Normal' if classification.get('class_name') == 'Normal' else 'Abnormal',
            'action': 'Consult cardiologist' if classification.get('class_name') != 'Normal' else 'No action needed',
            'follow_up': 'Routine follow-up' if classification.get('class_name') == 'Normal' else 'Follow-up as recommended'
        }
        
        generator.generate(
            signal=signal,
            binary_result={'class_name': 'Normal'},  # Placeholder
            multiclass_result=classification,
            ecg_metrics=ecg_metrics,
            metrics_status=data.get('metrics_status', {}),
            waveforms=waveforms,
            recommendation=recommendation,
            patient_id=patient_id,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_sex=patient_sex,
            doctor_name=doctor_name,
            indication=indication
        )
        
        # ---- Return PDF ----
        with open(report_path, 'rb') as f:
            pdf_data = f.read()
        
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{report_filename}"'
        
        logger.info(f"✓ Report generated and sent: {report_filename}")
        
        # Clean up
        try:
            report_path.unlink()
        except:
            pass
        
        return response
        
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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
    
    # Railway uses PORT environment variable, fallback to 8080 for local
    port = int(os.environ.get('PORT', 8080))
    host = '0.0.0.0'
    
    # Get local IP for network access display
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "YOUR_LOCAL_IP"
    
    logger.info(f"Server: http://127.0.0.1:{port}")
    logger.info(f"Network: http://{local_ip}:{port}")
    logger.info("Supported formats: " + ", ".join(ALLOWED_EXTENSIONS))
    logger.info("═" * 60)

    # Use Flask with threading for stability
    app.run(debug=False, host=host, port=port, threaded=True)
