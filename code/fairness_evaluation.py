"""
Comprehensive bias detection and fairness evaluation framework.

This module provides tools for:
- Group-wise performance evaluation across demographic groups
- Statistical significance testing for bias detection
- Fairness diagnostic visualizations
- Real-time bias monitoring during training
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


@dataclass
class GroupMetrics:
    """Container for group-wise performance metrics."""
    group_name: str
    group_value: Any
    n_samples: int
    mae: float
    rmse: float
    mean_prediction: float
    mean_true: float
    std_error: float
    confidence_interval: Tuple[float, float]


@dataclass
class FairnessReport:
    """Container for comprehensive fairness evaluation results."""
    overall_metrics: Dict[str, float]
    group_metrics: List[GroupMetrics]
    statistical_tests: Dict[str, Dict[str, float]]
    fairness_violations: List[str]
    bias_score: float


class GroupwiseEvaluator:
    """
    Implements group-wise performance evaluation with statistical significance testing.
    
    This class provides methods to:
    - Calculate MAE per demographic group (skin tone, device, etc.)
    - Perform statistical significance tests for group differences
    - Generate confidence intervals for group metrics
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the evaluator.
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.alpha = alpha
    
    def calculate_group_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        groups: np.ndarray,
        group_name: str
    ) -> List[GroupMetrics]:
        """
        Calculate performance metrics for each group.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            groups: Group labels for each sample
            group_name: Name of the grouping variable
            
        Returns:
            List of GroupMetrics objects, one per group
        """
        group_metrics = []
        unique_groups = np.unique(groups)
        
        for group_val in unique_groups:
            if pd.isna(group_val):
                continue
                
            mask = (groups == group_val)
            if np.sum(mask) == 0:
                continue
                
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Calculate metrics
            mae = mean_absolute_error(y_true_group, y_pred_group)
            rmse = np.sqrt(mean_squared_error(y_true_group, y_pred_group))
            
            # Calculate confidence interval for MAE using bootstrap
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                y_true_group, y_pred_group, metric='mae'
            )
            
            # Standard error of the mean absolute error
            residuals = np.abs(y_true_group - y_pred_group)
            std_error = np.std(residuals) / np.sqrt(len(residuals))
            
            group_metrics.append(GroupMetrics(
                group_name=f"{group_name}_{group_val}",
                group_value=group_val,
                n_samples=len(y_true_group),
                mae=mae,
                rmse=rmse,
                mean_prediction=np.mean(y_pred_group),
                mean_true=np.mean(y_true_group),
                std_error=std_error,
                confidence_interval=(ci_lower, ci_upper)
            ))
        
        return group_metrics
    
    def _bootstrap_confidence_interval(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metric: str = 'mae',
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metric: Metric to calculate ('mae' or 'rmse')
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        bootstrap_metrics = []
        n_samples = len(y_true)
        
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
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        return ci_lower, ci_upper
    
    def test_group_differences(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        groups: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform statistical tests for significant differences between groups.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            groups: Group labels for each sample
            
        Returns:
            Dictionary with test statistics and p-values
        """
        unique_groups = np.unique(groups)
        unique_groups = unique_groups[~pd.isna(unique_groups)]
        
        if len(unique_groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        # Calculate residuals for each group
        group_residuals = []
        group_maes = []
        
        for group_val in unique_groups:
            mask = (groups == group_val)
            if np.sum(mask) == 0:
                continue
                
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            residuals = y_true_group - y_pred_group
            mae = mean_absolute_error(y_true_group, y_pred_group)
            
            group_residuals.append(residuals)
            group_maes.append(mae)
        
        results = {}
        
        # Kruskal-Wallis test for differences in residual distributions
        if len(group_residuals) >= 2:
            try:
                kw_stat, kw_pval = stats.kruskal(*group_residuals)
                results['kruskal_wallis_stat'] = kw_stat
                results['kruskal_wallis_pval'] = kw_pval
            except Exception as e:
                results['kruskal_wallis_error'] = str(e)
        
        # Levene's test for equality of variances
        if len(group_residuals) >= 2:
            try:
                levene_stat, levene_pval = stats.levene(*group_residuals)
                results['levene_stat'] = levene_stat
                results['levene_pval'] = levene_pval
            except Exception as e:
                results['levene_error'] = str(e)
        
        # Calculate coefficient of variation for MAEs
        if len(group_maes) >= 2:
            mae_mean = np.mean(group_maes)
            mae_std = np.std(group_maes)
            results['mae_coefficient_variation'] = mae_std / mae_mean if mae_mean > 0 else 0
            results['mae_range'] = np.max(group_maes) - np.min(group_maes)
            results['mae_max_ratio'] = np.max(group_maes) / np.min(group_maes) if np.min(group_maes) > 0 else float('inf')
        
        return results
    
    def evaluate_multiple_groups(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metadata: pd.DataFrame,
        group_columns: List[str]
    ) -> Dict[str, List[GroupMetrics]]:
        """
        Evaluate performance across multiple demographic groupings.
        
        Args:
            y_true: True hemoglobin values
            y_pred: Predicted hemoglobin values
            metadata: DataFrame with demographic information
            group_columns: List of column names to group by
            
        Returns:
            Dictionary mapping group column names to lists of GroupMetrics
        """
        results = {}
        
        for group_col in group_columns:
            if group_col not in metadata.columns:
                warnings.warn(f"Column '{group_col}' not found in metadata")
                continue
            
            groups = metadata[group_col].values
            group_metrics = self.calculate_group_metrics(
                y_true, y_pred, groups, group_col
            )
            results[group_col] = group_metrics
        
        return results


class BiasDetector:
    """
    Implements bias detection algorithms including demographic parity and equalized odds.
    """
    
    def __init__(self, fairness_threshold: float = 0.15):
        """
        Initialize bias detector.
        
        Args:
            fairness_threshold: Maximum allowed relative difference between groups (default: 15%)
        """
        self.fairness_threshold = fairness_threshold
    
    def demographic_parity_difference(
        self, 
        y_pred: np.ndarray, 
        groups: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate demographic parity difference.
        
        For regression, we measure the difference in mean predictions across groups.
        
        Args:
            y_pred: Predicted values
            groups: Group labels
            threshold: Optional threshold for binary classification of predictions
            
        Returns:
            Dictionary with parity metrics
        """
        unique_groups = np.unique(groups)
        unique_groups = unique_groups[~pd.isna(unique_groups)]
        
        if len(unique_groups) < 2:
            return {"error": "Need at least 2 groups for parity calculation"}
        
        group_means = []
        for group_val in unique_groups:
            mask = (groups == group_val)
            if np.sum(mask) > 0:
                group_mean = np.mean(y_pred[mask])
                group_means.append(group_mean)
        
        if len(group_means) < 2:
            return {"error": "Insufficient groups with data"}
        
        # Calculate parity metrics
        overall_mean = np.mean(y_pred)
        max_deviation = max(abs(gm - overall_mean) for gm in group_means)
        relative_max_deviation = max_deviation / overall_mean if overall_mean != 0 else 0
        
        return {
            "demographic_parity_difference": max_deviation,
            "relative_demographic_parity_difference": relative_max_deviation,
            "group_means": group_means,
            "overall_mean": overall_mean
        }
    
    def equalized_odds_difference(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        groups: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate equalized odds difference for regression.
        
        We measure the difference in MAE across groups as a proxy for equalized odds.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            groups: Group labels
            
        Returns:
            Dictionary with equalized odds metrics
        """
        unique_groups = np.unique(groups)
        unique_groups = unique_groups[~pd.isna(unique_groups)]
        
        if len(unique_groups) < 2:
            return {"error": "Need at least 2 groups for equalized odds calculation"}
        
        group_maes = []
        for group_val in unique_groups:
            mask = (groups == group_val)
            if np.sum(mask) > 0:
                group_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                group_maes.append(group_mae)
        
        if len(group_maes) < 2:
            return {"error": "Insufficient groups with data"}
        
        # Calculate equalized odds metrics
        overall_mae = mean_absolute_error(y_true, y_pred)
        max_mae_deviation = max(abs(gm - overall_mae) for gm in group_maes)
        relative_max_mae_deviation = max_mae_deviation / overall_mae if overall_mae != 0 else 0
        
        return {
            "equalized_odds_difference": max_mae_deviation,
            "relative_equalized_odds_difference": relative_max_mae_deviation,
            "group_maes": group_maes,
            "overall_mae": overall_mae
        }
    
    def check_fairness_constraints(
        self, 
        group_metrics: List[GroupMetrics]
    ) -> List[str]:
        """
        Check if fairness constraints are violated.
        
        Args:
            group_metrics: List of GroupMetrics objects
            
        Returns:
            List of violation descriptions
        """
        violations = []
        
        if len(group_metrics) < 2:
            return violations
        
        # Extract MAE values
        maes = [gm.mae for gm in group_metrics]
        overall_mae = np.mean(maes)
        
        # Check relative MAE variation
        max_mae = max(maes)
        min_mae = min(maes)
        
        if min_mae > 0:
            relative_variation = (max_mae - min_mae) / min_mae
            if relative_variation > self.fairness_threshold:
                violations.append(
                    f"MAE variation ({relative_variation:.3f}) exceeds threshold ({self.fairness_threshold})"
                )
        
        # Check individual group deviations
        for gm in group_metrics:
            if overall_mae > 0:
                relative_deviation = abs(gm.mae - overall_mae) / overall_mae
                if relative_deviation > self.fairness_threshold:
                    violations.append(
                        f"Group {gm.group_name} MAE deviation ({relative_deviation:.3f}) exceeds threshold"
                    )
        
        return violations


def evaluate_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    group_columns: List[str] = None,
    fairness_threshold: float = 0.15
) -> FairnessReport:
    """
    Comprehensive fairness evaluation function.
    
    Args:
        y_true: True hemoglobin values
        y_pred: Predicted hemoglobin values
        metadata: DataFrame with demographic information
        group_columns: List of columns to evaluate (default: ['skin_tone_proxy', 'device_brand'])
        fairness_threshold: Maximum allowed relative difference between groups
        
    Returns:
        FairnessReport object with comprehensive results
    """
    if group_columns is None:
        group_columns = ['skin_tone_proxy', 'device_brand']
    
    # Initialize evaluators
    evaluator = GroupwiseEvaluator()
    bias_detector = BiasDetector(fairness_threshold)
    
    # Overall metrics
    overall_metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'n_samples': len(y_true)
    }
    
    # Group-wise evaluation
    all_group_metrics = []
    statistical_tests = {}
    
    for group_col in group_columns:
        if group_col not in metadata.columns:
            continue
        
        groups = metadata[group_col].values
        group_metrics = evaluator.calculate_group_metrics(
            y_true, y_pred, groups, group_col
        )
        all_group_metrics.extend(group_metrics)
        
        # Statistical tests
        test_results = evaluator.test_group_differences(y_true, y_pred, groups)
        statistical_tests[group_col] = test_results
    
    # Check fairness violations
    fairness_violations = bias_detector.check_fairness_constraints(all_group_metrics)
    
    # Calculate overall bias score (coefficient of variation of group MAEs)
    if all_group_metrics:
        group_maes = [gm.mae for gm in all_group_metrics]
        bias_score = np.std(group_maes) / np.mean(group_maes) if np.mean(group_maes) > 0 else 0
    else:
        bias_score = 0
    
    return FairnessReport(
        overall_metrics=overall_metrics,
        group_metrics=all_group_metrics,
        statistical_tests=statistical_tests,
        fairness_violations=fairness_violations,
        bias_score=bias_score
    )


if __name__ == "__main__":
    # Example usage
    print("Fairness evaluation module loaded successfully")
    print("Use evaluate_fairness() function for comprehensive bias detection")