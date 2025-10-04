#!/usr/bin/env python3
"""
ULTIMATE HACKATHON-WINNING MODEL
Super simple, bulletproof, and highly effective approach
Focus on what actually matters for hemoglobin estimation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class UltimateFeatureExtractor:
    """Ultra-focused feature extraction for hemoglobin"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_hemoglobin_features(self, image_path):
        """Extract only the most important features for HgB estimation"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((150, 150), Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 255.0
            
            features = []
            
            # 1. Core RGB Statistics (21 features)
            for channel in range(3):
                data = img_array[:, :, channel].flatten()
                features.extend([
                    np.mean(data),           # Mean intensity
                    np.std(data),            # Standard deviation
                    np.median(data),         # Median
                    np.percentile(data, 25), # Q1
                    np.percentile(data, 75), # Q3
                    np.min(data),            # Minimum
                    np.max(data)             # Maximum
                ])
            
            # 2. Critical Color Ratios (15 features)
            mean_r = np.mean(img_array[:, :, 0])
            mean_g = np.mean(img_array[:, :, 1])
            mean_b = np.mean(img_array[:, :, 2])
            
            eps = 1e-8
            features.extend([
                mean_r / (mean_g + eps),                    # R/G ratio (critical for HgB)
                mean_r / (mean_b + eps),                    # R/B ratio
                mean_g / (mean_b + eps),                    # G/B ratio
                mean_r / (mean_g + mean_b + eps),          # R/(G+B)
                (mean_r - mean_g) / (mean_r + mean_g + eps), # Normalized R-G
                (mean_r - mean_b) / (mean_r + mean_b + eps), # Normalized R-B
                mean_r * mean_g / (mean_b + eps),          # R*G/B
                (mean_r + mean_g) / (mean_b + eps),        # (R+G)/B
                mean_r / (mean_r + mean_g + mean_b + eps), # R proportion
                mean_g / (mean_r + mean_g + mean_b + eps), # G proportion
                mean_b / (mean_r + mean_g + mean_b + eps), # B proportion
                abs(mean_r - mean_g),                       # |R-G|
                abs(mean_r - mean_b),                       # |R-B|
                abs(mean_g - mean_b),                       # |G-B|
                max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b) # Range
            ])
            
            # 3. Simple Histogram Features (24 features - 8 bins per channel)
            for channel in range(3):
                hist, _ = np.histogram(img_array[:, :, channel], bins=8, range=(0, 1))
                hist = hist.astype(float) / (hist.sum() + eps)
                features.extend(hist)
            
            # 4. Regional Analysis (20 features)
            h, w = img_array.shape[:2]
            
            # Divide into 4 quadrants
            regions = [
                img_array[:h//2, :w//2],      # Top-left
                img_array[:h//2, w//2:],      # Top-right
                img_array[h//2:, :w//2],      # Bottom-left
                img_array[h//2:, w//2:]       # Bottom-right
            ]
            
            for region in regions:
                if region.size > 0:
                    features.extend([
                        np.mean(region[:, :, 0]),  # Mean R
                        np.mean(region[:, :, 1]),  # Mean G
                        np.mean(region[:, :, 2]),  # Mean B
                        np.std(region),            # Overall std
                        np.mean(region)            # Overall mean
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
            
            # 5. Simple Texture (10 features)
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray),
                np.percentile(gray, 90),
                np.percentile(gray, 10),
                np.var(gray),
                np.mean(gray > np.mean(gray)),  # Bright pixel ratio
                np.mean(gray < np.mean(gray))   # Dark pixel ratio
            ])
            
            # 6. Enhanced Color Analysis (10 features)
            # Focus on red channel variations (most important for HgB)
            red_channel = img_array[:, :, 0]
            
            features.extend([
                np.mean(red_channel > 0.5),                    # High red ratio
                np.mean(red_channel < 0.3),                    # Low red ratio
                np.std(red_channel) / (np.mean(red_channel) + eps), # Red CV
                np.mean(red_channel > np.percentile(red_channel, 75)), # Top quartile red
                np.mean(red_channel < np.percentile(red_channel, 25)), # Bottom quartile red
                np.max(red_channel) - np.min(red_channel),     # Red range
                np.mean(np.abs(red_channel - np.mean(red_channel))), # Red MAD
                np.corrcoef(red_channel.flatten(), img_array[:, :, 1].flatten())[0, 1], # R-G correlation
                np.corrcoef(red_channel.flatten(), img_array[:, :, 2].flatten())[0, 1], # R-B correlation
                np.mean(red_channel) / (np.mean(img_array) + eps) # Red dominance
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Feature extraction error for {image_path}: {e}")
            return np.zeros(100, dtype=np.float32)
    
    def fit_transform(self, image_paths):
        """Extract and scale features"""
        print("Extracting ultimate features...")
        features = []
        
        for i, path in enumerate(image_paths):
            feat = self.extract_hemoglobin_features(path)
            features.append(feat)
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(image_paths)}")
        
        X = np.array(features)
        print(f"Feature matrix: {X.shape}")
        
        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        
        return X_scaled
    
    def transform(self, image_paths):
        """Transform new images"""
        if not self.fitted:
            raise ValueError("Must fit first")
        
        features = []
        for path in image_paths:
            feat = self.extract_hemoglobin_features(path)
            features.append(feat)
        
        X = np.array(features)
        X = np.nan_to_num(X, nan=0.0)
        return self.scaler.transform(X)


class UltimateEnsemble:
    """Optimized ensemble for maximum performance"""
    
    def __init__(self):
        # Best performing models for this task
        self.models = {
            'ridge_strong': Ridge(alpha=0.01),  # Less regularization
            'ridge_medium': Ridge(alpha=0.1),
            'ridge_weak': Ridge(alpha=1.0),
            'elastic_balanced': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000),
            'elastic_l1': ElasticNet(alpha=0.1, l1_ratio=0.8, max_iter=3000),
            'rf_deep': RandomForestRegressor(
                n_estimators=500, 
                max_depth=12, 
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            ),
            'rf_wide': RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='log2',
                random_state=43
            ),
            'gbm_aggressive': GradientBoostingRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            'gbm_conservative': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                random_state=43
            )
        }
        
        self.weights = None
        self.fitted = False
    
    def fit(self, X, y):
        """Train ensemble with smart weighting"""
        print("Training ultimate ensemble...")
        
        # Train all models
        successful_models = {}
        for name, model in self.models.items():
            try:
                print(f"  Training {name}...")
                model.fit(X, y)
                successful_models[name] = model
            except Exception as e:
                print(f"    Failed: {e}")
        
        self.models = successful_models
        
        # Calculate performance-based weights
        self._calculate_smart_weights(X, y)
        self.fitted = True
        
        print(f"Successfully trained {len(self.models)} models")
    
    def _calculate_smart_weights(self, X, y):
        """Calculate weights based on individual model performance"""
        if len(X) < 5:
            # Not enough data for CV, use equal weights
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            print("  Using equal weights (insufficient data for CV)")
            return
        
        kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        model_scores = {name: [] for name in self.models.keys()}
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for name, model in self.models.items():
                try:
                    from sklearn.base import clone
                    temp_model = clone(model)
                    temp_model.fit(X_train, y_train)
                    pred = temp_model.predict(X_val)
                    mae = mean_absolute_error(y_val, pred)
                    # Convert MAE to score (lower MAE = higher score)
                    score = 1.0 / (mae + 0.5)
                    model_scores[name].append(score)
                except Exception as e:
                    print(f"    CV failed for {name}: {e}")
                    model_scores[name].append(0.1)
        
        # Calculate final weights
        avg_scores = {}
        for name, scores in model_scores.items():
            if scores:
                avg_scores[name] = np.mean(scores)
            else:
                avg_scores[name] = 0.1
        
        # Normalize weights
        total_score = sum(avg_scores.values())
        if total_score > 0:
            self.weights = {name: score/total_score for name, score in avg_scores.items()}
        else:
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        print("  Model weights:")
        for name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {weight:.3f}")
    
    def predict(self, X):
        """Make weighted ensemble prediction"""
        if not self.fitted:
            raise ValueError("Must fit ensemble first")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights[name])
            except Exception as e:
                print(f"Prediction failed for {name}: {e}")
        
        if not predictions:
            return np.full(len(X), 12.0)  # Default HgB
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimation"""
        predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except:
                predictions.append(np.full(len(X), 12.0))
        
        if not predictions:
            return np.full(len(X), 12.0), np.full(len(X), 2.0), np.full(len(X), 70.0)
        
        # Ensemble prediction
        ensemble_pred = self.predict(X)
        
        # Uncertainty as std of predictions
        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0)
        
        # Confidence based on agreement
        confidence = np.clip(90 - uncertainty * 10, 60, 95)
        
        return ensemble_pred, uncertainty, confidence


def load_data():
    """Load training data"""
    print("Loading data...")
    
    image_dir = Path("../1. Randomised Files")
    images = []
    labels = {}
    
    for img_file in image_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Test if image can be opened
                Image.open(img_file).convert('RGB')
                images.append(str(img_file))
                
                # Extract HgB value
                filename = img_file.name
                if 'HgB_' in filename and 'gdl' in filename:
                    try:
                        parts = filename.split('gdl')[0].split('_')
                        for part in reversed(parts):
                            try:
                                hgb = float(part)
                                if 4 <= hgb <= 25:
                                    labels[str(img_file)] = hgb
                                    break
                            except:
                                continue
                    except:
                        pass
            except:
                pass
    
    # Get labeled images
    labeled_images = [img for img in images if img in labels]
    y = np.array([labels[img] for img in labeled_images])
    
    print(f"Valid images: {len(images)}")
    print(f"Labeled images: {len(labeled_images)}")
    print(f"HgB range: {np.min(y):.1f} - {np.max(y):.1f}")
    print(f"HgB mean: {np.mean(y):.1f} ± {np.std(y):.1f}")
    
    return labeled_images, y


def train_ultimate_model():
    """Train the ultimate winning model"""
    print("🚀 ULTIMATE HACKATHON MODEL TRAINING 🚀")
    print("=" * 45)
    
    # Load data
    image_paths, y = load_data()
    
    if len(image_paths) < 3:
        print("❌ Need more training data")
        return
    
    # Extract features
    print("\n🎯 Extracting Key Features...")
    extractor = UltimateFeatureExtractor()
    X = extractor.fit_transform(image_paths)
    
    print(f"✅ Features: {X.shape}")
    
    # Train ensemble
    print("\n🧠 Training Ultimate Ensemble...")
    ensemble = UltimateEnsemble()
    ensemble.fit(X, y)
    
    # Evaluate
    print("\n📊 Performance Evaluation...")
    
    if len(X) >= 5:
        kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            temp_ensemble = UltimateEnsemble()
            temp_ensemble.fit(X_train, y_train)
            pred = temp_ensemble.predict(X_val)
            
            mae = mean_absolute_error(y_val, pred)
            cv_scores.append(mae)
            print(f"  Fold {fold+1}: MAE = {mae:.3f}")
        
        mean_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)
    else:
        # Use training error for small datasets
        pred = ensemble.predict(X)
        mean_mae = mean_absolute_error(y, pred)
        std_mae = 0.0
        print(f"  Training MAE: {mean_mae:.3f}")
    
    print(f"\n🎯 FINAL PERFORMANCE:")
    print(f"   MAE: {mean_mae:.3f} ± {std_mae:.3f}")
    
    if mean_mae < 1.0:
        grade = "🏆 OUTSTANDING"
    elif mean_mae < 2.0:
        grade = "🥇 EXCELLENT"
    elif mean_mae < 3.0:
        grade = "🥈 VERY GOOD"
    else:
        grade = "🥉 GOOD"
    
    print(f"   Grade: {grade}")
    
    # Save model
    print("\n💾 Saving Ultimate Model...")
    
    with open('../weights/ultimate_extractor.pkl', 'wb') as f:
        pickle.dump(extractor, f)
    
    with open('../weights/ultimate_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    
    metadata = {
        'model_type': 'Ultimate Hackathon Winner',
        'algorithms': list(ensemble.models.keys()),
        'feature_count': X.shape[1],
        'training_samples': len(X),
        'mae': float(mean_mae),
        'mae_std': float(std_mae),
        'weights': {k: float(v) for k, v in ensemble.weights.items()},
        'target_stats': {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'range': [float(np.min(y)), float(np.max(y))]
        },
        'version': '4.0.0-ultimate',
        'grade': grade.split()[1]
    }
    
    with open('../weights/ultimate_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ ULTIMATE MODEL SAVED!")
    print(f"🎯 Performance: {mean_mae:.3f} MAE")
    print("🏆 HACKATHON READY! 🏆")
    
    return ensemble, extractor, metadata


if __name__ == '__main__':
    train_ultimate_model()