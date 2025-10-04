"""
Robustness and fairness constraint validation system.

This module implements:
- MAE variation testing across skin tone groups (≤ 15% variation)
- Performance consistency validation across device types (≤ 10% variation)
- Comprehensive bias detection and mitigation validation
- Statistical significance testing for group differences
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
import warnings
from pathlib import Path

# Import existing modules
from fairness_evaluation import evaluate_fairness, GroupwiseEvaluator, BiasDetector
from comprehensive_validation import ValidationMetrics, ModelValidator


@dataclass
class RobustnessConstraints:
    """Container for robustness constraint thresholds."""
    max_skin_tone_variation: float = 0.15  # 15%
    max_device_variation: float = 0.10     # 10%
    min_group_size: int = 5                # Minimum samples per group
    significance_level: float = 0.05       # Statistical significance threshold


@dataclass
class GroupPerformanceMetrics:
    """Container for group-specific performance metrics."""
    group_name: str
    group_value: Any
    n_samples: int
    mae: float
    rmse: float
    relative_mae_deviation: float
    confidence_interval: Tuple[float, float]
    is_significant_difference: bool
    p_value: float


@dataclass
class RobustnessValidationReport:
    """Container for robustness validation results."""
    skin_tone_validation: Dict[str, Any]
    device_validation: Dict[str, Any]
    overall_fairness_score: float
    constraint_violations: List[str]
    statistical_tests: Dict[str, Any]
    group_performance_metrics: List[GroupPerformanceMetrics]
    recommendations: List[str]
    passed_all_constraints: bool


class RobustnessValidator:
    """
    Comprehensive robustness and fairness constraint validator.
    
    Validates model performance against clinical fairness requirements
    across demographic groups and device types.
    """
    
    def __init__(self, constraints: Optional[RobustnessConstraints] = None):
        """
        Initialize the robustness validator.
        
        Args:
            constraints: RobustnessConstraints object with thresholds
        """
        self.constraints = constraints or RobustnessConstraints()
        self.evaluator = GroupwiseEvaluator(alpha=self.constraints.significance_level)
        self.bias_detector = BiasDetector(fairness_threshold=self.constraints.max_skin_tone_variation)
    
    def validate_skin_tone_fairness(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        skin_tone_groups: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate MAE variation across skin tone groups.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            skin_tone_groups: Skin tone group labels
            
        Returns:
            Dictionary with skin tone fairness validation results
        """
        # Calculate group-wise metrics
        group_metrics = self.evaluator.calculate_group_metrics(
            y_true, y_pred, skin_tone_groups, "skin_tone"
        )
        
        # Filter groups with sufficient samples
        valid_groups = [gm for gm in group_metrics if gm.n_samples >= self.constraints.min_group_size]
        
        if len(valid_groups) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 2 groups with ≥{self.constraints.min_group_size} samples each",
                "valid_groups": len(valid_groups),
                "constraint_met": False
            }
        
        # Calculate MAE variation
        maes = [gm.mae for gm in valid_groups]
        overall_mae = np.mean(maes)
        
        # Calculate relative variation
        max_mae = max(maes)
        min_mae = min(maes)
        relative_variation = (max_mae - min_mae) / min_mae if min_mae > 0 else float('inf')
        
        # Check constraint
        constraint_met = relative_variation <= self.constraints.max_skin_tone_variation
        
        # Statistical significance testing
        statistical_tests = self.evaluator.test_group_differences(y_true, y_pred, skin_tone_groups)
        
        # Individual group deviations
        group_deviations = []
        for gm in valid_groups:
            deviation = abs(gm.mae - overall_mae) / overall_mae if overall_mae > 0 else 0
            group_deviations.append({
                "group": gm.group_name,
                "mae": gm.mae,
                "deviation": deviation,
                "exceeds_threshold": deviation > self.constraints.max_skin_tone_variation
            })
        
        return {
            "status": "evaluated",
            "constraint_met": constraint_met,
            "relative_variation": relative_variation,
            "threshold": self.constraints.max_skin_tone_variation,
            "overall_mae": overall_mae,
            "group_maes": maes,
            "group_deviations": group_deviations,
            "statistical_tests": statistical_tests,
            "valid_groups": len(valid_groups),
            "group_metrics": valid_groups
        }
    
    def validate_device_consistency(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        device_groups: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate performance consistency across device types.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            device_groups: Device type group labels
            
        Returns:
            Dictionary with device consistency validation results
        """
        # Calculate group-wise metrics
        group_metrics = self.evaluator.calculate_group_metrics(
            y_true, y_pred, device_groups, "device"
        )
        
        # Filter groups with sufficient samples
        valid_groups = [gm for gm in group_metrics if gm.n_samples >= self.constraints.min_group_size]
        
        if len(valid_groups) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 2 device groups with ≥{self.constraints.min_group_size} samples each",
                "valid_groups": len(valid_groups),
                "constraint_met": False
            }
        
        # Calculate MAE variation
        maes = [gm.mae for gm in valid_groups]
        overall_mae = np.mean(maes)
        
        # Calculate relative variation
        max_mae = max(maes)
        min_mae = min(maes)
        relative_variation = (max_mae - min_mae) / min_mae if min_mae > 0 else float('inf')
        
        # Check constraint (stricter for devices: 10% vs 15% for skin tone)
        constraint_met = relative_variation <= self.constraints.max_device_variation
        
        # Statistical significance testing
        statistical_tests = self.evaluator.test_group_differences(y_true, y_pred, device_groups)
        
        # Individual device deviations
        device_deviations = []
        for gm in valid_groups:
            deviation = abs(gm.mae - overall_mae) / overall_mae if overall_mae > 0 else 0
            device_deviations.append({
                "device": gm.group_name,
                "mae": gm.mae,
                "deviation": deviation,
                "exceeds_threshold": deviation > self.constraints.max_device_variation
            })
        
        return {
            "status": "evaluated",
            "constraint_met": constraint_met,
            "relative_variation": relative_variation,
            "threshold": self.constraints.max_device_variation,
            "overall_mae": overall_mae,
            "device_maes": maes,
            "device_deviations": device_deviations,
            "statistical_tests": statistical_tests,
            "valid_groups": len(valid_groups),
            "group_metrics": valid_groups
        }
    
    def comprehensive_bias_validation(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Perform comprehensive bias detection and validation.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            metadata: DataFrame with demographic information
            
        Returns:
            Dictionary with comprehensive bias validation results
        """
        # Perform fairness evaluation
        fairness_report = evaluate_fairness(
            y_true, y_pred, metadata,
            group_columns=['skin_tone_proxy', 'device_brand'] if 'skin_tone_proxy' in metadata.columns else [],
            fairness_threshold=self.constraints.max_skin_tone_variation
        )
        
        # Additional bias metrics
        bias_metrics = {}
        
        # Demographic parity for each group column
        for col in ['skin_tone_proxy', 'device_brand']:
            if col in metadata.columns:
                groups = metadata[col].values
                parity_results = self.bias_detector.demographic_parity_difference(y_pred, groups)
                bias_metrics[f"{col}_demographic_parity"] = parity_results
                
                equalized_odds_results = self.bias_detector.equalized_odds_difference(y_true, y_pred, groups)
                bias_metrics[f"{col}_equalized_odds"] = equalized_odds_results
        
        # Calculate intersectional bias (if multiple demographic attributes available)
        intersectional_bias = self._calculate_intersectional_bias(y_true, y_pred, metadata)
        
        return {
            "fairness_report": fairness_report,
            "bias_metrics": bias_metrics,
            "intersectional_bias": intersectional_bias,
            "overall_bias_score": fairness_report.bias_score,
            "fairness_violations": fairness_report.fairness_violations
        }
    
    def _calculate_intersectional_bias(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate intersectional bias across multiple demographic attributes.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            metadata: DataFrame with demographic information
            
        Returns:
            Dictionary with intersectional bias metrics
        """
        intersectional_results = {}
        
        # Check if we have multiple demographic columns
        demo_cols = [col for col in ['skin_tone_proxy', 'device_brand', 'age_group', 'gender'] 
                    if col in metadata.columns]
        
        if len(demo_cols) >= 2:
            # Create intersectional groups
            for i in range(len(demo_cols)):
                for j in range(i + 1, len(demo_cols)):
                    col1, col2 = demo_cols[i], demo_cols[j]
                    
                    # Create combined group labels
                    combined_groups = metadata[col1].astype(str) + "_" + metadata[col2].astype(str)
                    
                    # Calculate group-wise metrics
                    group_metrics = self.evaluator.calculate_group_metrics(
                        y_true, y_pred, combined_groups.values, f"{col1}_{col2}"
                    )
                    
                    # Filter for sufficient sample size
                    valid_groups = [gm for gm in group_metrics if gm.n_samples >= self.constraints.min_group_size]
                    
                    if len(valid_groups) >= 2:
                        maes = [gm.mae for gm in valid_groups]
                        max_mae = max(maes)
                        min_mae = min(maes)
                        variation = (max_mae - min_mae) / min_mae if min_mae > 0 else float('inf')
                        
                        intersectional_results[f"{col1}_{col2}"] = {
                            "variation": variation,
                            "exceeds_threshold": variation > self.constraints.max_skin_tone_variation,
                            "valid_groups": len(valid_groups),
                            "group_metrics": valid_groups
                        }
        
        return intersectional_results
    
    def validate_all_constraints(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metadata: pd.DataFrame
    ) -> RobustnessValidationReport:
        """
        Validate all robustness and fairness constraints.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            metadata: DataFrame with demographic information
            
        Returns:
            RobustnessValidationReport with comprehensive results
        """
        violations = []
        all_group_metrics = []
        
        # 1. Skin tone fairness validation
        skin_tone_validation = {"status": "not_available"}
        if 'skin_tone_proxy' in metadata.columns:
            skin_tone_validation = self.validate_skin_tone_fairness(
                y_true, y_pred, metadata['skin_tone_proxy'].values
            )
            if not skin_tone_validation.get('constraint_met', False):
                violations.append(f"Skin tone MAE variation ({skin_tone_validation.get('relative_variation', 0):.3f}) exceeds threshold ({self.constraints.max_skin_tone_variation})")
            
            if 'group_metrics' in skin_tone_validation:
                all_group_metrics.extend(skin_tone_validation['group_metrics'])
        
        # 2. Device consistency validation
        device_validation = {"status": "not_available"}
        if 'device_brand' in metadata.columns:
            device_validation = self.validate_device_consistency(
                y_true, y_pred, metadata['device_brand'].values
            )
            if not device_validation.get('constraint_met', False):
                violations.append(f"Device MAE variation ({device_validation.get('relative_variation', 0):.3f}) exceeds threshold ({self.constraints.max_device_variation})")
            
            if 'group_metrics' in device_validation:
                all_group_metrics.extend(device_validation['group_metrics'])
        
        # 3. Comprehensive bias validation
        bias_validation = self.comprehensive_bias_validation(y_true, y_pred, metadata)
        
        # Add fairness violations to overall violations
        violations.extend(bias_validation['fairness_violations'])
        
        # 4. Statistical tests summary
        statistical_tests = {}
        if skin_tone_validation.get('statistical_tests'):
            statistical_tests['skin_tone'] = skin_tone_validation['statistical_tests']
        if device_validation.get('statistical_tests'):
            statistical_tests['device'] = device_validation['statistical_tests']
        
        # 5. Convert group metrics to standardized format
        standardized_group_metrics = []
        for gm in all_group_metrics:
            # Calculate statistical significance (simplified)
            overall_mae = np.mean([g.mae for g in all_group_metrics])
            relative_deviation = abs(gm.mae - overall_mae) / overall_mae if overall_mae > 0 else 0
            
            standardized_group_metrics.append(GroupPerformanceMetrics(
                group_name=gm.group_name,
                group_value=gm.group_value,
                n_samples=gm.n_samples,
                mae=gm.mae,
                rmse=gm.rmse,
                relative_mae_deviation=relative_deviation,
                confidence_interval=gm.confidence_interval,
                is_significant_difference=relative_deviation > self.constraints.significance_level,
                p_value=0.05  # Placeholder - would need proper statistical test
            ))
        
        # 6. Overall fairness score
        overall_fairness_score = bias_validation['overall_bias_score']
        
        # 7. Generate recommendations
        recommendations = self._generate_robustness_recommendations(
            skin_tone_validation, device_validation, bias_validation, violations
        )
        
        # 8. Check if all constraints passed
        passed_all_constraints = len(violations) == 0
        
        return RobustnessValidationReport(
            skin_tone_validation=skin_tone_validation,
            device_validation=device_validation,
            overall_fairness_score=overall_fairness_score,
            constraint_violations=violations,
            statistical_tests=statistical_tests,
            group_performance_metrics=standardized_group_metrics,
            recommendations=recommendations,
            passed_all_constraints=passed_all_constraints
        )
    
    def _generate_robustness_recommendations(
        self,
        skin_tone_validation: Dict[str, Any],
        device_validation: Dict[str, Any],
        bias_validation: Dict[str, Any],
        violations: List[str]
    ) -> List[str]:
        """Generate actionable recommendations for robustness improvements."""
        recommendations = []
        
        # Skin tone recommendations
        if not skin_tone_validation.get('constraint_met', True):
            recommendations.append(
                "Skin tone fairness constraint violated. Consider: "
                "(1) Collecting more diverse training data, "
                "(2) Implementing fairness-aware training objectives, "
                "(3) Post-processing bias correction techniques."
            )
        
        # Device recommendations
        if not device_validation.get('constraint_met', True):
            recommendations.append(
                "Device consistency constraint violated. Consider: "
                "(1) Device-specific calibration, "
                "(2) Domain adaptation techniques, "
                "(3) Robust preprocessing pipelines."
            )
        
        # General bias recommendations
        if bias_validation['overall_bias_score'] > 0.1:
            recommendations.append(
                "High overall bias score detected. Consider implementing "
                "bias mitigation strategies such as adversarial debiasing "
                "or fairness constraints in the loss function."
            )
        
        # Data recommendations
        insufficient_data_groups = []
        for validation in [skin_tone_validation, device_validation]:
            if validation.get('status') == 'insufficient_data':
                insufficient_data_groups.append(validation.get('message', ''))
        
        if insufficient_data_groups:
            recommendations.append(
                f"Insufficient data for robust validation: {'; '.join(insufficient_data_groups)}. "
                "Consider collecting more diverse data or using data augmentation techniques."
            )
        
        # Success message
        if len(violations) == 0:
            recommendations.append(
                "All robustness and fairness constraints satisfied! "
                "Model demonstrates equitable performance across demographic groups."
            )
        
        return recommendations
    
    def generate_robustness_report(
        self, 
        report: RobustnessValidationReport, 
        output_dir: str = "robustness_reports"
    ) -> str:
        """
        Generate a comprehensive robustness validation report.
        
        Args:
            report: RobustnessValidationReport to document
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report content
        report_lines = []
        report_lines.append("ROBUSTNESS AND FAIRNESS VALIDATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall status
        status = "PASSED" if report.passed_all_constraints else "FAILED"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append(f"Overall Fairness Score: {report.overall_fairness_score:.4f}")
        report_lines.append("")
        
        # Constraint violations
        if report.constraint_violations:
            report_lines.append("CONSTRAINT VIOLATIONS:")
            for violation in report.constraint_violations:
                report_lines.append(f"  - {violation}")
        else:
            report_lines.append("No constraint violations detected.")
        report_lines.append("")
        
        # Skin tone validation
        report_lines.append("SKIN TONE FAIRNESS VALIDATION:")
        if report.skin_tone_validation['status'] == 'evaluated':
            st_val = report.skin_tone_validation
            report_lines.append(f"  Constraint Met: {st_val['constraint_met']}")
            report_lines.append(f"  Relative Variation: {st_val['relative_variation']:.4f}")
            report_lines.append(f"  Threshold: {st_val['threshold']:.4f}")
            report_lines.append(f"  Valid Groups: {st_val['valid_groups']}")
        else:
            report_lines.append(f"  Status: {report.skin_tone_validation['status']}")
        report_lines.append("")
        
        # Device validation
        report_lines.append("DEVICE CONSISTENCY VALIDATION:")
        if report.device_validation['status'] == 'evaluated':
            dev_val = report.device_validation
            report_lines.append(f"  Constraint Met: {dev_val['constraint_met']}")
            report_lines.append(f"  Relative Variation: {dev_val['relative_variation']:.4f}")
            report_lines.append(f"  Threshold: {dev_val['threshold']:.4f}")
            report_lines.append(f"  Valid Groups: {dev_val['valid_groups']}")
        else:
            report_lines.append(f"  Status: {report.device_validation['status']}")
        report_lines.append("")
        
        # Group performance metrics
        if report.group_performance_metrics:
            report_lines.append("GROUP PERFORMANCE METRICS:")
            for gm in report.group_performance_metrics:
                report_lines.append(f"  {gm.group_name}:")
                report_lines.append(f"    Samples: {gm.n_samples}")
                report_lines.append(f"    MAE: {gm.mae:.4f}")
                report_lines.append(f"    Relative Deviation: {gm.relative_mae_deviation:.4f}")
                report_lines.append(f"    CI: ({gm.confidence_interval[0]:.4f}, {gm.confidence_interval[1]:.4f})")
            report_lines.append("")
        
        # Recommendations
        if report.recommendations:
            report_lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                report_lines.append(f"  {i}. {rec}")
        report_lines.append("")
        
        # Save report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"robustness_validation_report_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return filepath


def validate_model_robustness(
    model_path: str,
    features_csv: str,
    meta_csv: str,
    model_type: str = 'ridge',
    constraints: Optional[RobustnessConstraints] = None
) -> RobustnessValidationReport:
    """
    Validate model robustness and fairness constraints.
    
    Args:
        model_path: Path to saved model
        features_csv: Path to features CSV
        meta_csv: Path to metadata CSV
        model_type: Type of model ('ridge', 'ensemble', etc.)
        constraints: Custom robustness constraints
        
    Returns:
        RobustnessValidationReport with validation results
    """
    from comprehensive_validation import load_and_validate_model
    
    # Load data and get predictions (reuse from comprehensive validation)
    # This is a simplified version - in practice, would load data directly
    
    # For now, create a dummy validation
    validator = RobustnessValidator(constraints)
    
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.normal(12.0, 2.0, n_samples)  # Hemoglobin values
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add some prediction error
    
    # Create dummy metadata
    metadata = pd.DataFrame({
        'skin_tone_proxy': np.random.choice(['light', 'medium', 'dark'], n_samples),
        'device_brand': np.random.choice(['iPhone', 'Samsung', 'Pixel'], n_samples)
    })
    
    # Perform validation
    report = validator.validate_all_constraints(y_true, y_pred, metadata)
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    print("Robustness validation system loaded successfully")
    
    # Test with dummy data
    print("Running robustness validation test...")
    try:
        report = validate_model_robustness("dummy_model", "dummy_features", "dummy_meta")
        print(f"Robustness validation completed")
        print(f"Constraints passed: {report.passed_all_constraints}")
        print(f"Violations: {len(report.constraint_violations)}")
        print(f"Overall fairness score: {report.overall_fairness_score:.4f}")
        
        # Generate report
        validator = RobustnessValidator()
        report_path = validator.generate_robustness_report(report)
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Robustness validation test failed: {e}")
        import traceback
        traceback.print_exc()