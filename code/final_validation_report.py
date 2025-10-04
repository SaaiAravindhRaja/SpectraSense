"""
Final performance and validation report generator.

This module implements:
- Comprehensive performance benchmarking report generation
- Fairness and bias evaluation documentation
- Deployment readiness assessment and recommendations
- Executive summary and technical documentation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import warnings

# Import existing modules
from comprehensive_validation import ComprehensiveValidationReport, ModelValidator
from robustness_validator import RobustnessValidationReport, RobustnessValidator
from performance_optimizer import OptimizationReport, ModelOptimizer
from robustness_tester import RobustnessTestReport, RobustnessTester
from fairness_evaluation import FairnessReport


@dataclass
class DeploymentReadinessAssessment:
    """Container for deployment readiness evaluation."""
    overall_readiness_score: float
    accuracy_readiness: bool
    performance_readiness: bool
    fairness_readiness: bool
    robustness_readiness: bool
    optimization_readiness: bool
    
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    estimated_deployment_timeline: str
    required_actions: List[str]
    risk_assessment: Dict[str, str]


@dataclass
class ExecutiveSummary:
    """Container for executive summary information."""
    project_name: str
    evaluation_date: str
    model_version: str
    
    key_achievements: List[str]
    performance_highlights: Dict[str, str]
    fairness_highlights: Dict[str, str]
    
    deployment_recommendation: str
    business_impact: str
    next_steps: List[str]


@dataclass
class FinalValidationReport:
    """Container for the complete final validation report."""
    executive_summary: ExecutiveSummary
    comprehensive_validation: ComprehensiveValidationReport
    robustness_validation: RobustnessValidationReport
    performance_optimization: OptimizationReport
    robustness_testing: RobustnessTestReport
    deployment_assessment: DeploymentReadinessAssessment
    
    report_metadata: Dict[str, Any]
    appendices: Dict[str, Any]


class ReportGenerator:
    """
    Comprehensive report generation system.
    
    Generates professional validation reports for stakeholders.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.model_validator = ModelValidator()
        self.robustness_validator = RobustnessValidator()
        self.performance_optimizer = ModelOptimizer()
        self.robustness_tester = RobustnessTester()
    
    def generate_executive_summary(
        self,
        validation_report: ComprehensiveValidationReport,
        robustness_report: RobustnessValidationReport,
        optimization_report: OptimizationReport,
        testing_report: RobustnessTestReport,
        project_name: str = "Hemoglobin Estimation Pipeline"
    ) -> ExecutiveSummary:
        """
        Generate executive summary for stakeholders.
        
        Args:
            validation_report: Comprehensive validation results
            robustness_report: Robustness validation results
            optimization_report: Performance optimization results
            testing_report: Robustness testing results
            project_name: Name of the project
            
        Returns:
            ExecutiveSummary with key findings and recommendations
        """
        # Key achievements
        achievements = []
        
        if validation_report.target_achievements.get('mae_target_achieved', False):
            mae = validation_report.validation_metrics.mae
            achievements.append(f"Achieved MAE target: {mae:.3f} g/dL (≤ 0.8 g/dL)")
        
        if validation_report.target_achievements.get('fairness_constraints_met', False):
            achievements.append("Met all fairness constraints across demographic groups")
        
        if optimization_report.all_targets_met:
            achievements.append("Met all performance optimization targets for production deployment")
        
        if testing_report.passed_all_tests:
            achievements.append("Passed comprehensive robustness testing across all conditions")
        
        # Performance highlights
        performance_highlights = {
            "Accuracy (MAE)": f"{validation_report.validation_metrics.mae:.3f} g/dL",
            "Inference Time": f"{optimization_report.optimized_metrics.inference_time_mean:.3f} seconds",
            "Model Size": f"{optimization_report.optimized_metrics.model_size_mb:.1f} MB",
            "Throughput": f"{testing_report.stress_test_result.average_throughput:.0f} images/minute"
        }
        
        # Fairness highlights
        fairness_highlights = {
            "Overall Bias Score": f"{validation_report.fairness_report.bias_score:.3f}",
            "Fairness Violations": str(len(validation_report.fairness_report.fairness_violations)),
            "Group Performance Consistency": "Excellent" if robustness_report.passed_all_constraints else "Needs Improvement"
        }
        
        # Deployment recommendation
        all_ready = (
            validation_report.target_achievements.get('mae_target_achieved', False) and
            validation_report.target_achievements.get('fairness_constraints_met', False) and
            optimization_report.all_targets_met and
            testing_report.passed_all_tests
        )
        
        if all_ready:
            deployment_rec = "READY FOR PRODUCTION DEPLOYMENT"
            business_impact = "High confidence for clinical deployment with expected accuracy and fairness standards met."
        elif len(achievements) >= 2:
            deployment_rec = "READY FOR PILOT DEPLOYMENT"
            business_impact = "Suitable for controlled pilot testing with monitoring and validation protocols."
        else:
            deployment_rec = "REQUIRES ADDITIONAL DEVELOPMENT"
            business_impact = "Additional optimization needed before deployment consideration."
        
        # Next steps
        next_steps = []
        if not all_ready:
            if not validation_report.target_achievements.get('mae_target_achieved', False):
                next_steps.append("Improve model accuracy through additional training or architecture changes")
            if not validation_report.target_achievements.get('fairness_constraints_met', False):
                next_steps.append("Address fairness violations through bias mitigation techniques")
            if not optimization_report.all_targets_met:
                next_steps.append("Complete performance optimization for production requirements")
            if not testing_report.passed_all_tests:
                next_steps.append("Resolve robustness issues identified in testing")
        else:
            next_steps.extend([
                "Prepare production deployment infrastructure",
                "Establish monitoring and alerting systems",
                "Conduct final user acceptance testing"
            ])
        
        return ExecutiveSummary(
            project_name=project_name,
            evaluation_date=datetime.now().strftime("%Y-%m-%d"),
            model_version="Enhanced Pipeline v1.0",
            key_achievements=achievements,
            performance_highlights=performance_highlights,
            fairness_highlights=fairness_highlights,
            deployment_recommendation=deployment_rec,
            business_impact=business_impact,
            next_steps=next_steps
        )
    
    def assess_deployment_readiness(
        self,
        validation_report: ComprehensiveValidationReport,
        robustness_report: RobustnessValidationReport,
        optimization_report: OptimizationReport,
        testing_report: RobustnessTestReport
    ) -> DeploymentReadinessAssessment:
        """
        Assess deployment readiness across all dimensions.
        
        Args:
            validation_report: Comprehensive validation results
            robustness_report: Robustness validation results
            optimization_report: Performance optimization results
            testing_report: Robustness testing results
            
        Returns:
            DeploymentReadinessAssessment with detailed readiness evaluation
        """
        # Individual readiness assessments
        accuracy_ready = validation_report.target_achievements.get('mae_target_achieved', False)
        performance_ready = optimization_report.all_targets_met
        fairness_ready = validation_report.target_achievements.get('fairness_constraints_met', False)
        robustness_ready = testing_report.passed_all_tests
        optimization_ready = optimization_report.all_targets_met
        
        # Overall readiness score (weighted average)
        readiness_weights = {
            'accuracy': 0.3,
            'performance': 0.2,
            'fairness': 0.2,
            'robustness': 0.2,
            'optimization': 0.1
        }
        
        readiness_scores = {
            'accuracy': 1.0 if accuracy_ready else 0.0,
            'performance': 1.0 if performance_ready else 0.5,
            'fairness': 1.0 if fairness_ready else 0.0,
            'robustness': testing_report.overall_robustness_score,
            'optimization': 1.0 if optimization_ready else 0.5
        }
        
        overall_score = sum(
            readiness_weights[key] * readiness_scores[key] 
            for key in readiness_weights
        )
        
        # Critical issues
        critical_issues = []
        if not accuracy_ready:
            mae = validation_report.validation_metrics.mae
            critical_issues.append(f"MAE ({mae:.3f} g/dL) exceeds clinical target (≤ 0.8 g/dL)")
        
        if not fairness_ready:
            violations = len(validation_report.fairness_report.fairness_violations)
            critical_issues.append(f"Fairness violations detected ({violations} constraint violations)")
        
        # Warnings
        warnings_list = []
        if not performance_ready:
            warnings_list.append("Performance targets not fully met - may impact user experience")
        
        if testing_report.overall_robustness_score < 0.8:
            warnings_list.append("Robustness score below recommended threshold (0.8)")
        
        # Recommendations
        recommendations = []
        recommendations.extend(validation_report.recommendations)
        recommendations.extend(robustness_report.recommendations)
        recommendations.extend(optimization_report.recommendations)
        recommendations.extend(testing_report.recommendations)
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        # Required actions
        required_actions = []
        if critical_issues:
            required_actions.append("Address all critical issues before deployment")
        if warnings_list:
            required_actions.append("Evaluate and mitigate warning conditions")
        if overall_score < 0.8:
            required_actions.append("Achieve minimum readiness score of 0.8")
        
        # Timeline estimation
        if overall_score >= 0.9:
            timeline = "Ready for immediate deployment"
        elif overall_score >= 0.7:
            timeline = "2-4 weeks additional development"
        elif overall_score >= 0.5:
            timeline = "1-2 months additional development"
        else:
            timeline = "3+ months additional development required"
        
        # Risk assessment
        risk_assessment = {
            "Technical Risk": "Low" if overall_score >= 0.8 else "Medium" if overall_score >= 0.6 else "High",
            "Clinical Risk": "Low" if accuracy_ready and fairness_ready else "High",
            "Performance Risk": "Low" if performance_ready else "Medium",
            "Regulatory Risk": "Low" if fairness_ready else "High"
        }
        
        return DeploymentReadinessAssessment(
            overall_readiness_score=overall_score,
            accuracy_readiness=accuracy_ready,
            performance_readiness=performance_ready,
            fairness_readiness=fairness_ready,
            robustness_readiness=robustness_ready,
            optimization_readiness=optimization_ready,
            critical_issues=critical_issues,
            warnings=warnings_list,
            recommendations=recommendations,
            estimated_deployment_timeline=timeline,
            required_actions=required_actions,
            risk_assessment=risk_assessment
        )
    
    def generate_comprehensive_report(
        self,
        model_name: str,
        validation_report: ComprehensiveValidationReport,
        robustness_report: RobustnessValidationReport,
        optimization_report: OptimizationReport,
        testing_report: RobustnessTestReport,
        project_name: str = "Hemoglobin Estimation Pipeline"
    ) -> FinalValidationReport:
        """
        Generate comprehensive final validation report.
        
        Args:
            model_name: Name of the model being validated
            validation_report: Comprehensive validation results
            robustness_report: Robustness validation results
            optimization_report: Performance optimization results
            testing_report: Robustness testing results
            project_name: Name of the project
            
        Returns:
            FinalValidationReport with complete validation documentation
        """
        # Generate executive summary
        executive_summary = self.generate_executive_summary(
            validation_report, robustness_report, optimization_report, testing_report, project_name
        )
        
        # Assess deployment readiness
        deployment_assessment = self.assess_deployment_readiness(
            validation_report, robustness_report, optimization_report, testing_report
        )
        
        # Report metadata
        report_metadata = {
            "report_version": "1.0",
            "generation_date": datetime.now().isoformat(),
            "model_name": model_name,
            "validation_framework_version": "1.0",
            "total_test_duration": "Comprehensive multi-phase validation",
            "validation_environment": "Development/Testing"
        }
        
        # Appendices with additional data
        appendices = {
            "detailed_metrics": {
                "accuracy_metrics": asdict(validation_report.validation_metrics),
                "calibration_metrics": asdict(validation_report.calibration_metrics),
                "performance_metrics": asdict(optimization_report.optimized_metrics)
            },
            "test_configurations": {
                "cross_validation_folds": 5,
                "bootstrap_samples": 1000,
                "stress_test_duration": testing_report.stress_test_result.test_duration
            },
            "statistical_tests": validation_report.fairness_report.statistical_tests
        }
        
        return FinalValidationReport(
            executive_summary=executive_summary,
            comprehensive_validation=validation_report,
            robustness_validation=robustness_report,
            performance_optimization=optimization_report,
            robustness_testing=testing_report,
            deployment_assessment=deployment_assessment,
            report_metadata=report_metadata,
            appendices=appendices
        )
    
    def save_report_as_json(
        self, 
        report: FinalValidationReport, 
        output_dir: str = "final_reports"
    ) -> str:
        """
        Save final report as JSON.
        
        Args:
            report: FinalValidationReport to save
            output_dir: Directory to save the report
            
        Returns:
            Path to the saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to dictionary with proper JSON serialization
        report_dict = asdict(report)
        
        # Handle numpy and other non-JSON serializable types
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_for_json(d)
        
        report_dict = recursive_convert(report_dict)
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_validation_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        return filepath
    
    def generate_markdown_report(
        self, 
        report: FinalValidationReport, 
        output_dir: str = "final_reports"
    ) -> str:
        """
        Generate human-readable markdown report.
        
        Args:
            report: FinalValidationReport to document
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated markdown file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate markdown content
        md_lines = []
        
        # Title and metadata
        md_lines.extend([
            f"# {report.executive_summary.project_name} - Final Validation Report",
            "",
            f"**Report Date:** {report.executive_summary.evaluation_date}",
            f"**Model Version:** {report.executive_summary.model_version}",
            f"**Report Version:** {report.report_metadata['report_version']}",
            "",
            "---",
            ""
        ])
        
        # Executive Summary
        md_lines.extend([
            "## Executive Summary",
            "",
            f"**Deployment Recommendation:** {report.executive_summary.deployment_recommendation}",
            "",
            f"**Business Impact:** {report.executive_summary.business_impact}",
            "",
            "### Key Achievements",
            ""
        ])
        
        for achievement in report.executive_summary.key_achievements:
            md_lines.append(f"- {achievement}")
        
        md_lines.extend(["", "### Performance Highlights", ""])
        for key, value in report.executive_summary.performance_highlights.items():
            md_lines.append(f"- **{key}:** {value}")
        
        md_lines.extend(["", "### Fairness Highlights", ""])
        for key, value in report.executive_summary.fairness_highlights.items():
            md_lines.append(f"- **{key}:** {value}")
        
        # Deployment Readiness
        md_lines.extend([
            "",
            "---",
            "",
            "## Deployment Readiness Assessment",
            "",
            f"**Overall Readiness Score:** {report.deployment_assessment.overall_readiness_score:.2f}/1.00",
            "",
            f"**Estimated Timeline:** {report.deployment_assessment.estimated_deployment_timeline}",
            "",
            "### Readiness by Category",
            "",
            f"- **Accuracy:** {'✅ Ready' if report.deployment_assessment.accuracy_readiness else '❌ Not Ready'}",
            f"- **Performance:** {'✅ Ready' if report.deployment_assessment.performance_readiness else '❌ Not Ready'}",
            f"- **Fairness:** {'✅ Ready' if report.deployment_assessment.fairness_readiness else '❌ Not Ready'}",
            f"- **Robustness:** {'✅ Ready' if report.deployment_assessment.robustness_readiness else '❌ Not Ready'}",
            ""
        ])
        
        # Critical Issues
        if report.deployment_assessment.critical_issues:
            md_lines.extend(["### Critical Issues", ""])
            for issue in report.deployment_assessment.critical_issues:
                md_lines.append(f"- ⚠️ {issue}")
            md_lines.append("")
        
        # Warnings
        if report.deployment_assessment.warnings:
            md_lines.extend(["### Warnings", ""])
            for warning in report.deployment_assessment.warnings:
                md_lines.append(f"- ⚡ {warning}")
            md_lines.append("")
        
        # Risk Assessment
        md_lines.extend(["### Risk Assessment", ""])
        for risk_type, risk_level in report.deployment_assessment.risk_assessment.items():
            emoji = "🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"
            md_lines.append(f"- **{risk_type}:** {emoji} {risk_level}")
        
        # Detailed Results
        md_lines.extend([
            "",
            "---",
            "",
            "## Detailed Validation Results",
            "",
            "### Accuracy Validation",
            "",
            f"- **MAE:** {report.comprehensive_validation.validation_metrics.mae:.4f} g/dL (Target: ≤ 0.8)",
            f"- **RMSE:** {report.comprehensive_validation.validation_metrics.rmse:.4f} g/dL",
            f"- **R²:** {report.comprehensive_validation.validation_metrics.r2:.4f}",
            f"- **Samples:** {report.comprehensive_validation.validation_metrics.n_samples}",
            "",
            "### Performance Optimization",
            "",
            f"- **Inference Time:** {report.performance_optimization.optimized_metrics.inference_time_mean:.3f}s (Target: ≤ 2.0s)",
            f"- **Model Size:** {report.performance_optimization.optimized_metrics.model_size_mb:.1f} MB (Target: ≤ 50 MB)",
            f"- **Memory Usage:** {report.performance_optimization.optimized_metrics.peak_memory_mb:.1f} MB (Target: ≤ 2048 MB)",
            f"- **Throughput:** {report.performance_optimization.optimized_metrics.throughput_images_per_minute:.0f} img/min (Target: ≥ 30)",
            "",
            "### Robustness Testing",
            "",
            f"- **Overall Robustness Score:** {report.robustness_testing.overall_robustness_score:.3f}",
            f"- **Performance Consistency:** {report.robustness_testing.performance_consistency_score:.3f}",
            f"- **Error Handling Score:** {report.robustness_testing.error_handling_score:.3f}",
            f"- **Stress Test Throughput:** {report.robustness_testing.stress_test_result.average_throughput:.0f} img/min",
            ""
        ])
        
        # Recommendations
        md_lines.extend([
            "---",
            "",
            "## Recommendations",
            ""
        ])
        
        all_recommendations = set()
        all_recommendations.update(report.comprehensive_validation.recommendations)
        all_recommendations.update(report.deployment_assessment.recommendations)
        
        for i, rec in enumerate(sorted(all_recommendations), 1):
            md_lines.append(f"{i}. {rec}")
        
        # Next Steps
        md_lines.extend([
            "",
            "## Next Steps",
            ""
        ])
        
        for i, step in enumerate(report.executive_summary.next_steps, 1):
            md_lines.append(f"{i}. {step}")
        
        # Footer
        md_lines.extend([
            "",
            "---",
            "",
            f"*Report generated on {report.report_metadata['generation_date']}*",
            f"*Validation Framework Version: {report.report_metadata['validation_framework_version']}*"
        ])
        
        # Save markdown report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_validation_report_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(md_lines))
        
        return filepath


def create_dummy_reports() -> Tuple[ComprehensiveValidationReport, RobustnessValidationReport, OptimizationReport, RobustnessTestReport]:
    """Create dummy reports for testing the final report generation."""
    
    # Import required classes
    from comprehensive_validation import ValidationMetrics, CalibrationMetrics, CrossValidationResults
    from robustness_validator import RobustnessConstraints
    from performance_optimizer import PerformanceMetrics, PerformanceTargets
    from robustness_tester import StressTestResult, EdgeCaseTestResult, ConditionTestResult
    from fairness_evaluation import FairnessReport, GroupMetrics
    
    # Create dummy validation report
    validation_metrics = ValidationMetrics(
        mae=0.75, rmse=0.95, mape=6.2, r2=0.85, n_samples=200,
        confidence_interval_mae=(0.70, 0.80), confidence_interval_rmse=(0.90, 1.00)
    )
    
    calibration_metrics = CalibrationMetrics(
        expected_calibration_error=0.04, maximum_calibration_error=0.08,
        reliability_diagram_data={}, calibration_slope=0.02, calibration_intercept=0.01, brier_score=0.15
    )
    
    cv_results = CrossValidationResults(
        cv_scores_mae=np.array([0.72, 0.78, 0.74, 0.76, 0.75]),
        cv_scores_rmse=np.array([0.92, 0.98, 0.94, 0.96, 0.95]),
        mean_cv_mae=0.75, std_cv_mae=0.02, mean_cv_rmse=0.95, std_cv_rmse=0.02,
        consistency_score=0.027, statistical_significance={'mae_vs_target_pvalue': 0.02, 'mae_significantly_better': True}
    )
    
    fairness_report = FairnessReport(
        overall_metrics={'mae': 0.75, 'rmse': 0.95, 'n_samples': 200},
        group_metrics=[], statistical_tests={}, fairness_violations=[], bias_score=0.08
    )
    
    comprehensive_validation = ComprehensiveValidationReport(
        model_name="enhanced_hemoglobin_model",
        validation_metrics=validation_metrics,
        calibration_metrics=calibration_metrics,
        cross_validation_results=cv_results,
        fairness_report=fairness_report,
        target_achievements={'mae_target_achieved': True, 'ece_target_achieved': True, 'fairness_constraints_met': True, 'cv_consistency_good': True},
        recommendations=["Model meets all accuracy and fairness targets"],
        timestamp=datetime.now().isoformat()
    )
    
    # Create dummy robustness validation report
    robustness_validation = RobustnessValidationReport(
        skin_tone_validation={'constraint_met': True, 'relative_variation': 0.12},
        device_validation={'constraint_met': True, 'relative_variation': 0.08},
        overall_fairness_score=0.08,
        constraint_violations=[],
        statistical_tests={},
        group_performance_metrics=[],
        recommendations=["All fairness constraints satisfied"],
        passed_all_constraints=True
    )
    
    # Create dummy optimization report
    original_metrics = PerformanceMetrics(
        inference_time_mean=3.2, inference_time_std=0.5, inference_time_p95=4.1,
        model_size_mb=75.0, peak_memory_mb=2500.0, throughput_images_per_minute=18.0,
        cpu_utilization_percent=85.0, meets_time_target=False, meets_size_target=False,
        meets_memory_target=False, meets_throughput_target=False
    )
    
    optimized_metrics = PerformanceMetrics(
        inference_time_mean=1.8, inference_time_std=0.3, inference_time_p95=2.2,
        model_size_mb=45.0, peak_memory_mb=1800.0, throughput_images_per_minute=35.0,
        cpu_utilization_percent=70.0, meets_time_target=True, meets_size_target=True,
        meets_memory_target=True, meets_throughput_target=True
    )
    
    optimization_report = OptimizationReport(
        original_metrics=original_metrics,
        optimized_metrics=optimized_metrics,
        optimization_techniques=['memory_optimization', 'batch_optimization', 'caching'],
        performance_improvement={'inference_time': -0.44, 'model_size': -0.40, 'memory_usage': -0.28, 'throughput': 0.94},
        recommendations=["All performance targets achieved"],
        all_targets_met=True
    )
    
    # Create dummy robustness testing report
    stress_result = StressTestResult(
        test_duration=60.0, total_images_processed=2100, successful_predictions=2095,
        failed_predictions=5, average_throughput=35.0, peak_throughput=42.0, min_throughput=28.0,
        memory_usage_peak=1850.0, cpu_usage_peak=75.0, meets_throughput_target=True, system_stability_score=0.92
    )
    
    robustness_testing = RobustnessTestReport(
        condition_test_results=[], edge_case_test_results=[], stress_test_result=stress_result,
        overall_robustness_score=0.89, performance_consistency_score=0.85, error_handling_score=0.92,
        recommendations=["Excellent robustness across all test conditions"], passed_all_tests=True
    )
    
    return comprehensive_validation, robustness_validation, optimization_report, robustness_testing


def generate_final_validation_report(
    model_name: str = "enhanced_hemoglobin_model",
    project_name: str = "Hemoglobin Estimation Pipeline Enhancement"
) -> FinalValidationReport:
    """
    Generate a comprehensive final validation report.
    
    Args:
        model_name: Name of the model being validated
        project_name: Name of the project
        
    Returns:
        FinalValidationReport with complete validation documentation
    """
    # Create dummy reports for demonstration
    validation_report, robustness_report, optimization_report, testing_report = create_dummy_reports()
    
    # Generate comprehensive report
    generator = ReportGenerator()
    final_report = generator.generate_comprehensive_report(
        model_name, validation_report, robustness_report, optimization_report, testing_report, project_name
    )
    
    return final_report


if __name__ == "__main__":
    # Example usage and testing
    print("Final validation report generator loaded successfully")
    
    # Generate comprehensive final report
    print("Generating final validation report...")
    try:
        final_report = generate_final_validation_report()
        
        print(f"Final report generated successfully")
        print(f"Deployment recommendation: {final_report.executive_summary.deployment_recommendation}")
        print(f"Overall readiness score: {final_report.deployment_assessment.overall_readiness_score:.2f}")
        print(f"Key achievements: {len(final_report.executive_summary.key_achievements)}")
        
        # Save reports
        generator = ReportGenerator()
        
        # Save JSON report
        json_path = generator.save_report_as_json(final_report)
        print(f"JSON report saved to: {json_path}")
        
        # Save Markdown report
        md_path = generator.generate_markdown_report(final_report)
        print(f"Markdown report saved to: {md_path}")
        
        print("\nExecutive Summary:")
        print(f"- Project: {final_report.executive_summary.project_name}")
        print(f"- Date: {final_report.executive_summary.evaluation_date}")
        print(f"- Recommendation: {final_report.executive_summary.deployment_recommendation}")
        print(f"- Business Impact: {final_report.executive_summary.business_impact}")
        
        print("\nNext Steps:")
        for i, step in enumerate(final_report.executive_summary.next_steps, 1):
            print(f"  {i}. {step}")
        
    except Exception as e:
        print(f"Final report generation failed: {e}")
        import traceback
        traceback.print_exc()