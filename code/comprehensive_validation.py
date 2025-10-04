"""
Comprehensive accuracy validation system for hemoglobin estimation pipeline.

This module implements:
- Test dataset evaluation with MAE target validation (≤ 0.8 g/dL)
- Cross-validation consistency checks and statistical significance testing
- Confidence calibration accuracy measurement (ECE ≤ 0.05)
- Performance validation across multiple model architectures
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import warnings
import pickle
import joblib
from pathlib import Path

# Import existing modules
from fairness_evaluation import evaluate_fairness, FairnessReport
from meta_encode import MetaEncoder, load_meta_csv


@dataclass
class ValidationMetrics:
    """Container for comprehensive validation metrics."""
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    r2: float
    n_samples: int
    confidence_interval_mae: Tuple[float, float]
    confidence_interval_rmse: Tuple[float, float]


@dataclass
class CalibrationMetrics:
    """Container for calibration assessment metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    reliability_diagram_data: Dict[str, np.ndarray]
    calibration_slope: float
    calibration_intercept: float
    brier_score: float


@dataclass
class CrossValidationResults:
    """Container for cross-validation results."""
    cv_scores_mae: np.ndarray
    cv_scores_rmse: np.ndarray
    mean_cv_mae: float
    std_cv_mae: float
    mean_cv_rmse: float
    std_cv_rmse: float
    consistency_score: float  # Coefficient of variation
    statistical_significance: Dict[str, float]


@dataclass
class ComprehensiveValidationReport:
    """Container for complete validation assessment."""
    model_name: str
    validation_metrics: ValidationMetrics
    calibration_metrics: CalibrationMetrics
    cross_validation_results: CrossValidationResults
    fairness_report: FairnessReport
    target_achievements: Dict[str, bool]
    recommendations: List[str]
    timestamp: str


class AccuracyValidator:
    """
    Comprehensive accuracy validation system.
    
    Implements rigorous testing protocols to validate model performance
    against clinical accuracy requirements.
    """
    
    def __init__(self, mae_target: float = 0.8, ece_target: float = 0.05):
        """
        Initialize the accuracy validator.
        
        Args:
            mae_target: Target MAE threshold (g/dL)
            ece_target: Target Expected Calibration Error threshold
        """
        self.mae_target = mae_target
        self.ece_target = ece_target
        
    def calculate_validation_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        confidence_level: float = 0.95
    ) -> ValidationMetrics:
        """
        Calculate comprehensive validation metrics with confidence intervals.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            confidence_level: Confidence level for intervals
            
        Returns:
            ValidationMetrics object with all computed metrics
        """
        n_samples = len(y_true)
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Bootstrap confidence intervals
        ci_mae = self._bootstrap_confidence_interval(
            y_true, y_pred, metric='mae', confidence_level=confidence_level
        )
        ci_rmse = self._bootstrap_confidence_interval(
            y_true, y_pred, metric='rmse', confidence_level=confidence_level
        )
        
        return ValidationMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            r2=r2,
            n_samples=n_samples,
            confidence_interval_mae=ci_mae,
            confidence_interval_rmse=ci_rmse
        )
    
    def _bootstrap_confidence_interval(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metric: str = 'mae',
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric: Metric to calculate ('mae' or 'rmse')
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        bootstrap_metrics = []
        n_samples = len(y_true)
        
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metric
            if metric == 'mae':
                boot_metric = mean_absolute_error(y_true_boot, y_pred_boot)
            elif metric == 'rmse':
                boot_metric = np.sqrt(mean_squared_error(y_true_boot, y_pred_boot))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            bootstrap_metrics.append(boot_metric)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        return ci_lower, ci_upper
    
    def assess_calibration(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None,
        n_bins: int = 10
    ) -> CalibrationMetrics:
        """
        Assess prediction calibration quality.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            y_pred_std: Predicted standard deviations (optional)
            n_bins: Number of bins for calibration assessment
            
        Returns:
            CalibrationMetrics object with calibration assessment
        """
        # If no uncertainty estimates provided, use residual-based approach
        if y_pred_std is None:
            # Estimate uncertainty from residuals
            residuals = np.abs(y_true - y_pred)
            y_pred_std = np.full_like(y_pred, np.std(residuals))
        
        # Calculate Expected Calibration Error (ECE)
        ece, mce, reliability_data = self._calculate_calibration_error(
            y_true, y_pred, y_pred_std, n_bins
        )
        
        # Fit calibration curve
        try:
            # Create binary targets based on prediction intervals
            confidence_levels = [0.68, 0.95]  # 1-sigma and 2-sigma
            calibration_slope = 0
            calibration_intercept = 0
            
            for conf_level in confidence_levels:
                z_score = stats.norm.ppf((1 + conf_level) / 2)
                lower_bound = y_pred - z_score * y_pred_std
                upper_bound = y_pred + z_score * y_pred_std
                
                # Check if true values fall within prediction intervals
                within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
                expected_coverage = conf_level
                actual_coverage = np.mean(within_interval)
                
                # Simple linear relationship for calibration
                calibration_slope += (actual_coverage - expected_coverage)
            
            calibration_slope /= len(confidence_levels)
            
        except Exception as e:
            warnings.warn(f"Calibration curve fitting failed: {e}")
            calibration_slope = 0
            calibration_intercept = 0
        
        # Calculate Brier score (adapted for regression)
        brier_score = np.mean((y_true - y_pred) ** 2)
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            reliability_diagram_data=reliability_data,
            calibration_slope=calibration_slope,
            calibration_intercept=calibration_intercept,
            brier_score=brier_score
        )
    
    def _calculate_calibration_error(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_std: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, float, Dict[str, np.ndarray]]:
        """
        Calculate Expected and Maximum Calibration Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_pred_std: Predicted standard deviations
            n_bins: Number of bins for calibration
            
        Returns:
            Tuple of (ECE, MCE, reliability_diagram_data)
        """
        # Create confidence scores based on prediction uncertainty
        confidence_scores = 1 / (1 + y_pred_std)  # Higher confidence = lower uncertainty
        
        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this confidence bin
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy in this bin (using MAE-based accuracy)
                bin_mae = mean_absolute_error(y_true[in_bin], y_pred[in_bin])
                # Convert MAE to accuracy-like metric (lower MAE = higher accuracy)
                max_possible_error = np.max(y_true) - np.min(y_true)
                bin_accuracy = 1 - (bin_mae / max_possible_error) if max_possible_error > 0 else 1
                
                # Average confidence in this bin
                bin_confidence = confidence_scores[in_bin].mean()
                
                # Calibration error for this bin
                bin_error = abs(bin_accuracy - bin_confidence)
                
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        reliability_data = {
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries
        }
        
        return ece, mce, reliability_data
    
    def perform_cross_validation(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        model,
        cv_folds: int = 5,
        random_state: int = 42
    ) -> CrossValidationResults:
        """
        Perform cross-validation with consistency checks.
        
        Args:
            X: Feature matrix
            y: Target values
            model: Model object with fit/predict methods
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            
        Returns:
            CrossValidationResults object with CV metrics
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        cv_mae_scores = []
        cv_rmse_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model
            model_clone = self._clone_model(model)
            model_clone.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model_clone.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            cv_mae_scores.append(mae)
            cv_rmse_scores.append(rmse)
        
        cv_mae_scores = np.array(cv_mae_scores)
        cv_rmse_scores = np.array(cv_rmse_scores)
        
        # Calculate consistency metrics
        consistency_score = np.std(cv_mae_scores) / np.mean(cv_mae_scores)
        
        # Statistical significance testing
        # Test if CV scores are significantly different from target
        mae_ttest = stats.ttest_1samp(cv_mae_scores, self.mae_target)
        
        statistical_significance = {
            'mae_vs_target_statistic': mae_ttest.statistic,
            'mae_vs_target_pvalue': mae_ttest.pvalue,
            'mae_significantly_better': (mae_ttest.pvalue < 0.05) and (np.mean(cv_mae_scores) < self.mae_target)
        }
        
        return CrossValidationResults(
            cv_scores_mae=cv_mae_scores,
            cv_scores_rmse=cv_rmse_scores,
            mean_cv_mae=np.mean(cv_mae_scores),
            std_cv_mae=np.std(cv_mae_scores),
            mean_cv_rmse=np.mean(cv_rmse_scores),
            std_cv_rmse=np.std(cv_rmse_scores),
            consistency_score=consistency_score,
            statistical_significance=statistical_significance
        )
    
    def _clone_model(self, model):
        """Clone a model for cross-validation."""
        try:
            # Try sklearn-style cloning
            from sklearn.base import clone
            return clone(model)
        except:
            # Fallback for custom models
            if hasattr(model, 'get_params') and hasattr(model, 'set_params'):
                params = model.get_params()
                new_model = type(model)(**params)
                return new_model
            else:
                # Simple copy for basic models
                import copy
                return copy.deepcopy(model)


class ModelValidator:
    """
    High-level model validation orchestrator.
    
    Coordinates comprehensive validation across multiple models and datasets.
    """
    
    def __init__(self, mae_target: float = 0.8, ece_target: float = 0.05):
        """
        Initialize the model validator.
        
        Args:
            mae_target: Target MAE threshold (g/dL)
            ece_target: Target Expected Calibration Error threshold
        """
        self.mae_target = mae_target
        self.ece_target = ece_target
        self.accuracy_validator = AccuracyValidator(mae_target, ece_target)
    
    def validate_model_comprehensive(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: pd.DataFrame,
        X_features: Optional[np.ndarray] = None,
        model_object = None,
        y_pred_std: Optional[np.ndarray] = None
    ) -> ComprehensiveValidationReport:
        """
        Perform comprehensive validation of a model.
        
        Args:
            model_name: Name/identifier of the model
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            metadata: DataFrame with demographic information
            X_features: Feature matrix (for cross-validation)
            model_object: Model object (for cross-validation)
            y_pred_std: Predicted standard deviations (optional)
            
        Returns:
            ComprehensiveValidationReport with all validation results
        """
        from datetime import datetime
        
        # 1. Basic accuracy validation
        validation_metrics = self.accuracy_validator.calculate_validation_metrics(
            y_true, y_pred
        )
        
        # 2. Calibration assessment
        calibration_metrics = self.accuracy_validator.assess_calibration(
            y_true, y_pred, y_pred_std
        )
        
        # 3. Cross-validation (if features and model provided)
        if X_features is not None and model_object is not None:
            cv_results = self.accuracy_validator.perform_cross_validation(
                X_features, y_true, model_object
            )
        else:
            # Create dummy CV results
            cv_results = CrossValidationResults(
                cv_scores_mae=np.array([validation_metrics.mae]),
                cv_scores_rmse=np.array([validation_metrics.rmse]),
                mean_cv_mae=validation_metrics.mae,
                std_cv_mae=0.0,
                mean_cv_rmse=validation_metrics.rmse,
                std_cv_rmse=0.0,
                consistency_score=0.0,
                statistical_significance={'mae_vs_target_pvalue': 1.0, 'mae_significantly_better': False}
            )
        
        # 4. Fairness evaluation
        fairness_report = evaluate_fairness(
            y_true, y_pred, metadata,
            group_columns=['skin_tone_proxy', 'device_brand'] if 'skin_tone_proxy' in metadata.columns else []
        )
        
        # 5. Check target achievements
        target_achievements = {
            'mae_target_achieved': validation_metrics.mae <= self.mae_target,
            'ece_target_achieved': calibration_metrics.expected_calibration_error <= self.ece_target,
            'fairness_constraints_met': len(fairness_report.fairness_violations) == 0,
            'cv_consistency_good': cv_results.consistency_score < 0.1  # CV < 10%
        }
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(
            validation_metrics, calibration_metrics, fairness_report, target_achievements
        )
        
        return ComprehensiveValidationReport(
            model_name=model_name,
            validation_metrics=validation_metrics,
            calibration_metrics=calibration_metrics,
            cross_validation_results=cv_results,
            fairness_report=fairness_report,
            target_achievements=target_achievements,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_recommendations(
        self,
        validation_metrics: ValidationMetrics,
        calibration_metrics: CalibrationMetrics,
        fairness_report: FairnessReport,
        target_achievements: Dict[str, bool]
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # MAE recommendations
        if not target_achievements['mae_target_achieved']:
            mae_gap = validation_metrics.mae - self.mae_target
            recommendations.append(
                f"MAE ({validation_metrics.mae:.3f}) exceeds target ({self.mae_target}). "
                f"Consider model architecture improvements or additional training data."
            )
        
        # Calibration recommendations
        if not target_achievements['ece_target_achieved']:
            recommendations.append(
                f"Calibration error ({calibration_metrics.expected_calibration_error:.3f}) "
                f"exceeds target ({self.ece_target}). Consider temperature scaling or "
                f"Platt scaling for better calibration."
            )
        
        # Fairness recommendations
        if not target_achievements['fairness_constraints_met']:
            recommendations.append(
                f"Fairness violations detected: {fairness_report.fairness_violations}. "
                f"Consider bias mitigation techniques or additional diverse training data."
            )
        
        # Cross-validation recommendations
        if not target_achievements['cv_consistency_good']:
            recommendations.append(
                "High cross-validation variance detected. Consider regularization "
                "techniques or ensemble methods for more stable predictions."
            )
        
        # Positive feedback
        if all(target_achievements.values()):
            recommendations.append(
                "All validation targets achieved! Model is ready for deployment consideration."
            )
        
        return recommendations
    
    def save_validation_report(
        self, 
        report: ComprehensiveValidationReport, 
        output_dir: str = "validation_reports"
    ) -> str:
        """
        Save validation report to disk.
        
        Args:
            report: ComprehensiveValidationReport to save
            output_dir: Directory to save reports
            
        Returns:
            Path to saved report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert report to dictionary for JSON serialization
        report_dict = asdict(report)
        
        # Handle numpy arrays in the report
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert numpy objects
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_numpy(d)
        
        report_dict = recursive_convert(report_dict)
        
        # Save as JSON
        timestamp = report.timestamp.replace(':', '-').replace('.', '-')
        filename = f"validation_report_{report.model_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        return filepath


def load_and_validate_model(
    model_path: str,
    features_csv: str,
    meta_csv: str,
    model_type: str = 'ridge'
) -> ComprehensiveValidationReport:
    """
    Load a saved model and perform comprehensive validation.
    
    Args:
        model_path: Path to saved model
        features_csv: Path to features CSV
        meta_csv: Path to metadata CSV
        model_type: Type of model ('ridge', 'ensemble', etc.)
        
    Returns:
        ComprehensiveValidationReport with validation results
    """
    # Load data
    df = pd.read_csv(features_csv, index_col=0)
    df['features'] = df['features'].apply(
        lambda c: np.array(json.loads(c)) if not pd.isna(c) else np.array([])
    )
    
    meta = load_meta_csv(meta_csv)
    
    # Align data
    common = df.index.intersection(meta.index.astype(str))
    df = df.loc[common]
    meta = meta.loc[common]
    
    X_img = np.vstack(df['features'].values)
    y = df['label'].values.astype(float)
    
    # Load metadata encoder and create full feature matrix
    encoder_path = os.path.join(os.path.dirname(model_path), 'meta_encoder.json')
    if os.path.exists(encoder_path):
        encoder = MetaEncoder.load(encoder_path)
        X_meta = encoder.transform(meta)
        X_all = np.hstack([X_img, X_meta])
    else:
        X_all = X_img
    
    # Deterministic split for validation
    rng = np.random.RandomState(1234)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(len(idx) * 0.8)
    val_idx = idx[cut:]
    
    X_val = X_all[val_idx]
    y_val = y[val_idx]
    meta_val = meta.iloc[val_idx]
    
    # Load and predict with model
    if model_type == 'ridge':
        model_data = np.load(model_path)
        coef = model_data['coef']
        intercept = float(model_data['intercept'])
        y_pred = X_val @ coef + intercept
        
        # Create dummy model object for CV
        from sklearn.linear_model import Ridge
        model_object = Ridge(alpha=1.0)
        model_object.coef_ = coef
        model_object.intercept_ = intercept
        
    elif model_type == 'ensemble':
        with open(model_path, 'rb') as f:
            ensemble = pickle.load(f)
        y_pred = ensemble.predict(X_val)
        model_object = ensemble
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform comprehensive validation
    validator = ModelValidator()
    model_name = os.path.basename(model_path).replace('.npz', '').replace('.pkl', '')
    
    report = validator.validate_model_comprehensive(
        model_name=model_name,
        y_true=y_val,
        y_pred=y_pred,
        metadata=meta_val,
        X_features=X_all,
        model_object=model_object
    )
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    print("Comprehensive validation system loaded successfully")
    
    # Test with existing model if available
    code_dir = os.path.dirname(__file__)
    model_path = os.path.join(code_dir, '..', 'weights', 'feature_meta_ridge.npz')
    features_csv = os.path.join(code_dir, 'image_features.csv')
    meta_csv = os.path.join(code_dir, 'sample_meta.csv')
    
    if all(os.path.exists(p) for p in [model_path, features_csv, meta_csv]):
        print("Running validation on existing model...")
        try:
            report = load_and_validate_model(model_path, features_csv, meta_csv, 'ridge')
            print(f"Validation completed for {report.model_name}")
            print(f"MAE: {report.validation_metrics.mae:.4f} (target: ≤ 0.8)")
            print(f"ECE: {report.calibration_metrics.expected_calibration_error:.4f} (target: ≤ 0.05)")
            print(f"Targets achieved: {sum(report.target_achievements.values())}/{len(report.target_achievements)}")
        except Exception as e:
            print(f"Validation failed: {e}")
    else:
        print("Required files not found for testing")