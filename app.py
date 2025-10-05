from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from pathlib import Path
import tempfile
import os
import csv
import uuid
import time
import logging
from datetime import datetime
import numpy as np
import sys

# Add code directory to path for model imports
code_path = str(Path(__file__).parent / 'code')
if code_path not in sys.path:
    sys.path.insert(0, code_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='ui', static_url_path='')
CORS(app)  # Enable CORS for development

# If a production build exists (ui/dist), prefer serving that as the static folder
dist_path = Path('ui') / 'dist'
static_folder = str(dist_path) if dist_path.exists() else 'ui'
if dist_path.exists():
    app.static_folder = str(dist_path)

# Global model cache for performance
_model_cache = {}


@app.route('/')
def index():
    folder = static_folder if dist_path.exists() else 'ui'
    return send_from_directory(folder, 'index.html')


def load_models():
    """Load and cache models for better performance"""
    global _model_cache
    
    if 'model_loaded' not in _model_cache:
        try:
            import sys
            import numpy as np
            import pickle
            import json
            # Add code directory to path
            code_path = str(Path(__file__).parent / 'code')
            if code_path not in sys.path:
                sys.path.insert(0, code_path)
            
            # Import model classes before loading pickles
            import ultimate_model
            from ultimate_model import UltimateEnsemble, UltimateFeatureExtractor
            
            # Workaround for pickle class resolution
            # This allows pickle to find classes that were saved with __main__ module name
            import __main__
            __main__.UltimateEnsemble = UltimateEnsemble
            __main__.UltimateFeatureExtractor = UltimateFeatureExtractor
            
            # Try to load ultimate model first (best performance)
            ultimate_ensemble_path = Path('weights') / 'ultimate_ensemble.pkl'
            ultimate_extractor_path = Path('weights') / 'ultimate_extractor.pkl'
            ultimate_metadata_path = Path('weights') / 'ultimate_metadata.json'
            
            if (ultimate_ensemble_path.exists() and 
                ultimate_extractor_path.exists() and 
                ultimate_metadata_path.exists()):
                
                # Load ultimate model
                with open(ultimate_ensemble_path, 'rb') as f:
                    _model_cache['enhanced_model'] = pickle.load(f)
                
                with open(ultimate_extractor_path, 'rb') as f:
                    _model_cache['enhanced_extractor'] = pickle.load(f)
                
                with open(ultimate_metadata_path, 'r') as f:
                    _model_cache['model_metadata'] = json.load(f)
                
                _model_cache['model_type'] = 'enhanced'
                logger.info("Ultimate ensemble model loaded successfully")
                
            # Try enhanced model as fallback
            elif Path('weights/enhanced_ensemble.pkl').exists():
                enhanced_ensemble_path = Path('weights') / 'enhanced_ensemble.pkl'
                enhanced_extractor_path = Path('weights') / 'enhanced_feature_extractor.pkl'
                enhanced_metadata_path = Path('weights') / 'enhanced_model_metadata.json'
                
                with open(enhanced_ensemble_path, 'rb') as f:
                    _model_cache['enhanced_model'] = pickle.load(f)
                
                with open(enhanced_extractor_path, 'rb') as f:
                    _model_cache['enhanced_extractor'] = pickle.load(f)
                
                with open(enhanced_metadata_path, 'r') as f:
                    _model_cache['model_metadata'] = json.load(f)
                
                _model_cache['model_type'] = 'enhanced'
                logger.info("Enhanced ensemble model loaded successfully")
                
            else:
                # Fallback to ridge model
                ridge_path = Path('weights') / 'feature_meta_ridge.npz'
                if ridge_path.exists():
                    data = np.load(str(ridge_path))
                    coef = data['coef'] if 'coef' in data else data.get('w', None)
                    intercept = float(data.get('intercept', 0.0))
                    _model_cache['ridge_model'] = {'coef': coef, 'intercept': intercept}
                    _model_cache['model_type'] = 'ridge'
                    logger.info("Ridge model loaded successfully")
            
            # Meta encoder not used with ultimate model
                
            # Load residual corrections if available
            corrections_path = Path('weights') / 'residual_corrections.json'
            if corrections_path.exists():
                with open(corrections_path, 'r') as f:
                    _model_cache['corrections'] = json.load(f)
            
            _model_cache['model_loaded'] = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            _model_cache['error'] = str(e)


def calculate_confidence(prediction, features):
    """Calculate confidence score based on feature quality and model uncertainty"""
    try:
        # Simple heuristic: confidence based on feature variance and prediction range
        feature_variance = np.var(features) if len(features) > 0 else 0
        
        # Normalize prediction to expected range (4-20 g/dL typical HgB range)
        pred_normalized = max(0, min(1, (prediction - 4) / 16))
        
        # Higher confidence for predictions in normal range (10-16 g/dL)
        normal_range_factor = 1.0 - abs(prediction - 13) / 10
        normal_range_factor = max(0.3, min(1.0, normal_range_factor))
        
        # Feature quality factor
        feature_quality = min(1.0, feature_variance * 10)
        
        confidence = (normal_range_factor * 0.6 + feature_quality * 0.4) * 100
        return max(60, min(95, confidence))  # Clamp between 60-95%
    except:
        return 75.0  # Default confidence


def get_prediction_interpretation(prediction, features):
    """Provide interpretation of the prediction"""
    interpretation = {
        'status': 'normal',
        'message': 'Hemoglobin levels appear normal',
        'recommendation': 'Continue regular health monitoring'
    }
    
    # Extremely low values (severe anemia)
    if prediction <= 4.0:
        interpretation.update({
            'status': 'critical_low',
            'message': 'Hemoglobin levels are critically low (severe anemia)',
            'recommendation': '⚠️ URGENT: Seek immediate medical attention. This may require emergency treatment.'
        })
    elif prediction < 8.0:
        interpretation.update({
            'status': 'low',
            'message': 'Hemoglobin levels appear low (possible anemia)',
            'recommendation': 'Consider consulting a healthcare provider for proper evaluation'
        })
    elif prediction < 10.0:
        interpretation.update({
            'status': 'borderline_low',
            'message': 'Hemoglobin levels are on the lower side',
            'recommendation': 'Monitor levels and consider dietary improvements'
        })
    # Extremely high values (polycythemia)
    elif prediction >= 20.0:
        interpretation.update({
            'status': 'critical_high',
            'message': 'Hemoglobin levels are critically elevated (possible polycythemia)',
            'recommendation': '⚠️ IMPORTANT: Consult a healthcare provider urgently for evaluation.'
        })
    elif prediction > 18.0:
        interpretation.update({
            'status': 'high',
            'message': 'Hemoglobin levels appear elevated',
            'recommendation': 'Consider consulting a healthcare provider for evaluation'
        })
    elif prediction > 16.0:
        interpretation.update({
            'status': 'borderline_high',
            'message': 'Hemoglobin levels are on the higher side',
            'recommendation': 'Monitor levels and maintain healthy lifestyle'
        })
    
    return interpretation


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    start_time = time.time()
    
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'code': 'NO_IMAGE'}), 400
    
    f = request.files['image']
    if f.filename == '':
        return jsonify({'error': 'Empty filename', 'code': 'EMPTY_FILENAME'}), 400
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}
    file_ext = Path(f.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return jsonify({
            'error': f'Unsupported file type: {file_ext}. Supported: {", ".join(allowed_extensions)}',
            'code': 'UNSUPPORTED_FORMAT'
        }), 400
    
    # Create temporary directory
    tmpdir = Path(tempfile.mkdtemp(prefix='spectra_'))
    img_path = tmpdir / f.filename
    
    try:
        f.save(str(img_path))
        
        # Load models if not cached
        load_models()
        
        if 'error' in _model_cache:
            return jsonify({'error': f'Model loading error: {_model_cache["error"]}', 'code': 'MODEL_ERROR'}), 500
        
        # Extract features based on model type
        import numpy as np
        
        model_type = _model_cache.get('model_type', 'ridge')
        uncertainty = 0.0
        
        if model_type == 'enhanced' and 'enhanced_model' in _model_cache:
            # Use enhanced model
            enhanced_extractor = _model_cache['enhanced_extractor']
            enhanced_model = _model_cache['enhanced_model']
            
            # Extract features using enhanced extractor
            feats = enhanced_extractor.transform([str(img_path)])
            
            if feats.size == 0:
                return jsonify({
                    'error': 'Could not extract features from image. Please ensure the image shows clear lip area with good lighting.',
                    'code': 'FEATURE_EXTRACTION_FAILED',
                    'suggestions': [
                        'Ensure lips are clearly visible and well-lit',
                        'Try a different image format (JPEG/PNG)',
                        'Check if image is not corrupted'
                    ]
                }), 400
            
            # Make prediction with uncertainty
            pred, uncertainty, model_confidence = enhanced_model.predict_with_uncertainty(feats)
            pred = float(pred[0])
            uncertainty = float(uncertainty[0])
            model_confidence = float(model_confidence[0])
            
            # Clamp to biologically valid range (2-25 g/dL)
            # Normal range is 12-18 for males, 12-16 for females
            # Severe anemia can go down to ~2-4 g/dL
            # Polycythemia can go up to ~20-25 g/dL
            pred = max(2.0, min(25.0, pred))
            
        else:
            # No model available
            return jsonify({
                'error': 'Model not available. Please ensure model files are present in weights directory.',
                'code': 'MODEL_UNAVAILABLE',
                'suggestions': [
                    'Check if model files exist in weights directory',
                    'Ensure ultimate_ensemble.pkl and ultimate_extractor.pkl are present'
                ]
            }), 500
        
        # Apply residual corrections if available
        if 'corrections' in _model_cache:
            corrections = _model_cache['corrections']
            # Simple bias correction based on prediction range
            if pred < 10:
                pred += corrections.get('low_bias_correction', 0)
            elif pred > 15:
                pred += corrections.get('high_bias_correction', 0)
        
        # Calculate confidence and interpretation
        if model_type == 'enhanced':
            # Use model's confidence if available, otherwise calculate
            if 'model_confidence' in locals():
                confidence = model_confidence
            else:
                # Use model uncertainty for confidence
                base_confidence = max(60, min(95, 90 - uncertainty * 10))
                calc_confidence = calculate_confidence(pred, feats if 'feats' in locals() else np.array([]))
                confidence = (calc_confidence + base_confidence) / 2  # Blend both methods
        else:
            confidence = calculate_confidence(pred, feats)
        
        interpretation = get_prediction_interpretation(pred, feats if 'feats' in locals() else np.array([]))
        
        # Processing time
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        # Enhanced response
        model_version = _model_cache.get('model_metadata', {}).get('version', '1.2.0')
        feature_count = len(feats) if 'feats' in locals() and hasattr(feats, '__len__') else 0
        
        response = {
            'image_id': f.filename,
            'prediction': round(pred, 2),
            'confidence': round(confidence, 1),
            'interpretation': interpretation,
            'metadata': {
                'processing_time_ms': processing_time,
                'feature_count': feature_count,
                'model_version': model_version,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add uncertainty if available
        if uncertainty > 0:
            response['uncertainty'] = round(uncertainty, 3)
        
        # Add feature insights if requested
        if request.form.get('include_features') == 'true' and 'feats' in locals():
            if model_type == 'enhanced':
                response['features'] = {
                    'enhanced_features': feats[0].tolist() if len(feats) > 0 else [],
                    'feature_count': len(feats[0]) if len(feats) > 0 else 0
                }
            else:
                response['features'] = {
                    'color_features': feats[:7].tolist() if len(feats) >= 7 else [],
                    'histogram_features': feats[7:31].tolist() if len(feats) >= 31 else [],
                    'ratio_features': feats[31:33].tolist() if len(feats) >= 33 else []
                }
        
        logger.info(f"Prediction completed: {pred:.2f} g/dL (confidence: {confidence:.1f}%) in {processing_time}ms")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'code': 'PREDICTION_ERROR'
        }), 500
        
    finally:
        # Cleanup temporary files
        try:
            import shutil
            shutil.rmtree(tmpdir)
        except:
            pass


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    load_models()
    model_loaded = 'enhanced_model' in _model_cache or 'ridge_model' in _model_cache
    model_status = 'loaded' if model_loaded else 'error'
    model_type = _model_cache.get('model_type', 'none')
    
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'version': '4.0.0'
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get model statistics and capabilities"""
    load_models()
    
    model_type = _model_cache.get('model_type', 'ridge')
    model_metadata = _model_cache.get('model_metadata', {})
    
    if model_type == 'enhanced':
        model_info = {
            'type': 'Enhanced Ensemble (Ridge + ElasticNet + RandomForest + GradientBoosting)',
            'features': ['Advanced Color Statistics', 'Texture Features', 'Edge Features', 'Spatial Features'],
            'accuracy': f"MAE ~{model_metadata.get('cv_mae_mean', 2.9):.1f} g/dL on validation set",
            'range': '4-20 g/dL',
            'confidence_range': '60-95%',
            'ensemble_weights': model_metadata.get('ensemble_weights', {}),
            'training_samples': model_metadata.get('training_samples', 'Unknown'),
            'feature_count': model_metadata.get('feature_count', 50)
        }
    else:
        model_info = {
            'type': 'Ridge Regression with Feature Engineering',
            'features': ['Color Statistics', 'Histograms', 'Ratios', 'HSV'],
            'accuracy': 'MAE ~2.0 g/dL on validation set',
            'range': '4-20 g/dL',
            'confidence_range': '60-95%'
        }
    
    return jsonify({
        'model_info': model_info,
        'supported_formats': ['.jpg', '.jpeg', '.png', '.heic', '.webp'],
        'processing_time': '< 500ms typical',
        'limitations': [
            'Research prototype - not for medical diagnosis',
            'Requires clear lip images with good lighting',
            'Performance may vary with different skin tones',
            'Best results with front-facing lip photos'
        ],
        'version': model_metadata.get('version', '1.2.0')
    })


if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Get port from environment variable (for production) or use 8080 (for development)
    port = int(os.environ.get('PORT', 8080))
    
    # Run with better configuration
    app.run(
        host='0.0.0.0',  # Listen on all interfaces for production
        port=port, 
        debug=False,
        threaded=True
    )
