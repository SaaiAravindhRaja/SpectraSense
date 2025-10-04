"""
Comprehensive robustness testing system.

This module implements:
- Performance testing across different lighting conditions and devices
- Edge case handling and error recovery mechanism validation
- Stress testing for high-throughput scenarios (≥ 30 images/minute)
- Comprehensive robustness evaluation across various conditions
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
import json
import pickle
import concurrent.futures
import threading
from pathlib import Path
import warnings
from contextlib import contextmanager
import traceback

# Import existing modules
from performance_optimizer import PerformanceProfiler, PerformanceTargets
from fairness_evaluation import evaluate_fairness


@dataclass
class RobustnessTestConditions:
    """Container for robustness testing conditions."""
    lighting_conditions: List[str] = None
    device_types: List[str] = None
    image_qualities: List[str] = None
    stress_test_duration: float = 60.0  # seconds
    min_throughput_target: float = 30.0  # images per minute
    error_tolerance: float = 0.05  # 5% error rate tolerance
    
    def __post_init__(self):
        if self.lighting_conditions is None:
            self.lighting_conditions = ['bright', 'normal', 'dim', 'artificial', 'outdoor']
        if self.device_types is None:
            self.device_types = ['iPhone', 'Samsung', 'Pixel', 'generic']
        if self.image_qualities is None:
            self.image_qualities = ['high', 'medium', 'low', 'compressed']


@dataclass
class ConditionTestResult:
    """Container for test results under specific conditions."""
    condition_name: str
    condition_value: str
    n_samples: int
    mae: float
    rmse: float
    inference_time_mean: float
    inference_time_std: float
    error_rate: float
    success_rate: float
    throughput: float
    meets_performance_target: bool
    meets_accuracy_target: bool


@dataclass
class EdgeCaseTestResult:
    """Container for edge case test results."""
    edge_case_name: str
    test_description: str
    n_tests: int
    success_count: int
    failure_count: int
    error_types: Dict[str, int]
    recovery_success_rate: float
    average_recovery_time: float


@dataclass
class StressTestResult:
    """Container for stress test results."""
    test_duration: float
    total_images_processed: int
    successful_predictions: int
    failed_predictions: int
    average_throughput: float
    peak_throughput: float
    min_throughput: float
    memory_usage_peak: float
    cpu_usage_peak: float
    meets_throughput_target: bool
    system_stability_score: float


@dataclass
class RobustnessTestReport:
    """Container for comprehensive robustness test results."""
    condition_test_results: List[ConditionTestResult]
    edge_case_test_results: List[EdgeCaseTestResult]
    stress_test_result: StressTestResult
    overall_robustness_score: float
    performance_consistency_score: float
    error_handling_score: float
    recommendations: List[str]
    passed_all_tests: bool


class ConditionSimulator:
    """
    Simulates various environmental and device conditions for testing.
    """
    
    def __init__(self):
        """Initialize the condition simulator."""
        pass
    
    def simulate_lighting_condition(self, image_data: np.ndarray, condition: str) -> np.ndarray:
        """
        Simulate different lighting conditions on image data.
        
        Args:
            image_data: Input image data
            condition: Lighting condition to simulate
            
        Returns:
            Modified image data simulating the condition
        """
        if not isinstance(image_data, np.ndarray):
            return image_data
        
        # Simulate lighting effects
        if condition == 'bright':
            # Increase brightness and reduce contrast
            return np.clip(image_data * 1.3 + 0.2, 0, 1)
        elif condition == 'dim':
            # Decrease brightness and increase noise
            noise = np.random.normal(0, 0.05, image_data.shape)
            return np.clip(image_data * 0.6 + noise, 0, 1)
        elif condition == 'artificial':
            # Add color cast (yellowish)
            modified = image_data.copy()
            if len(modified.shape) == 3 and modified.shape[2] >= 3:
                modified[:, :, 0] *= 1.1  # Increase red
                modified[:, :, 1] *= 1.05  # Slightly increase green
            return np.clip(modified, 0, 1)
        elif condition == 'outdoor':
            # Add blue cast and increase contrast
            modified = image_data.copy()
            if len(modified.shape) == 3 and modified.shape[2] >= 3:
                modified[:, :, 2] *= 1.1  # Increase blue
            return np.clip(modified * 1.2 - 0.1, 0, 1)
        else:  # normal
            return image_data
    
    def simulate_device_characteristics(self, image_data: np.ndarray, device: str) -> np.ndarray:
        """
        Simulate device-specific characteristics.
        
        Args:
            image_data: Input image data
            device: Device type to simulate
            
        Returns:
            Modified image data simulating device characteristics
        """
        if not isinstance(image_data, np.ndarray):
            return image_data
        
        # Simulate device effects
        if device == 'iPhone':
            # High quality, slight warm cast
            return np.clip(image_data * 1.02, 0, 1)
        elif device == 'Samsung':
            # Saturated colors
            return np.clip(image_data * 1.1 - 0.05, 0, 1)
        elif device == 'Pixel':
            # Neutral processing
            return image_data
        elif device == 'generic':
            # Lower quality, add noise
            noise = np.random.normal(0, 0.03, image_data.shape)
            return np.clip(image_data + noise, 0, 1)
        else:
            return image_data
    
    def simulate_image_quality(self, image_data: np.ndarray, quality: str) -> np.ndarray:
        """
        Simulate different image quality levels.
        
        Args:
            image_data: Input image data
            quality: Quality level to simulate
            
        Returns:
            Modified image data simulating the quality level
        """
        if not isinstance(image_data, np.ndarray):
            return image_data
        
        # Simulate quality effects
        if quality == 'high':
            return image_data
        elif quality == 'medium':
            # Slight blur and compression artifacts
            noise = np.random.normal(0, 0.02, image_data.shape)
            return np.clip(image_data + noise, 0, 1)
        elif quality == 'low':
            # More blur and noise
            noise = np.random.normal(0, 0.05, image_data.shape)
            return np.clip(image_data * 0.95 + noise, 0, 1)
        elif quality == 'compressed':
            # Heavy compression artifacts
            noise = np.random.normal(0, 0.08, image_data.shape)
            return np.clip(image_data * 0.9 + noise, 0, 1)
        else:
            return image_data


class EdgeCaseTester:
    """
    Tests model behavior on edge cases and error conditions.
    """
    
    def __init__(self):
        """Initialize the edge case tester."""
        self.condition_simulator = ConditionSimulator()
    
    def test_invalid_inputs(self, model_func: Callable) -> EdgeCaseTestResult:
        """Test model behavior with invalid inputs."""
        test_cases = [
            ("empty_array", np.array([])),
            ("nan_values", np.full((10,), np.nan)),
            ("inf_values", np.full((10,), np.inf)),
            ("negative_values", np.full((10,), -1.0)),
            ("very_large_values", np.full((10,), 1e10)),
            ("wrong_shape", np.random.randn(5, 5, 5)),
            ("none_input", None),
            ("string_input", "invalid"),
        ]
        
        success_count = 0
        failure_count = 0
        error_types = {}
        recovery_times = []
        
        for case_name, test_input in test_cases:
            try:
                start_time = time.time()
                result = model_func(test_input)
                recovery_time = time.time() - start_time
                recovery_times.append(recovery_time)
                success_count += 1
            except Exception as e:
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
                failure_count += 1
        
        recovery_success_rate = success_count / len(test_cases) if test_cases else 0
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        return EdgeCaseTestResult(
            edge_case_name="invalid_inputs",
            test_description="Testing model behavior with invalid input data",
            n_tests=len(test_cases),
            success_count=success_count,
            failure_count=failure_count,
            error_types=error_types,
            recovery_success_rate=recovery_success_rate,
            average_recovery_time=avg_recovery_time
        )
    
    def test_extreme_conditions(self, model_func: Callable, test_inputs: List[Any]) -> EdgeCaseTestResult:
        """Test model behavior under extreme conditions."""
        extreme_conditions = [
            ("extreme_bright", lambda x: self.condition_simulator.simulate_lighting_condition(x, 'bright')),
            ("extreme_dim", lambda x: self.condition_simulator.simulate_lighting_condition(x, 'dim')),
            ("high_noise", lambda x: x + np.random.normal(0, 0.2, x.shape) if isinstance(x, np.ndarray) else x),
            ("low_contrast", lambda x: x * 0.3 + 0.35 if isinstance(x, np.ndarray) else x),
            ("high_contrast", lambda x: np.clip((x - 0.5) * 3 + 0.5, 0, 1) if isinstance(x, np.ndarray) else x),
        ]
        
        success_count = 0
        failure_count = 0
        error_types = {}
        recovery_times = []
        
        for condition_name, condition_func in extreme_conditions:
            for test_input in test_inputs[:5]:  # Test with first 5 inputs
                try:
                    start_time = time.time()
                    modified_input = condition_func(test_input)
                    result = model_func(modified_input)
                    recovery_time = time.time() - start_time
                    recovery_times.append(recovery_time)
                    success_count += 1
                except Exception as e:
                    error_type = type(e).__name__
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    failure_count += 1
        
        total_tests = len(extreme_conditions) * min(5, len(test_inputs))
        recovery_success_rate = success_count / total_tests if total_tests > 0 else 0
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        return EdgeCaseTestResult(
            edge_case_name="extreme_conditions",
            test_description="Testing model behavior under extreme environmental conditions",
            n_tests=total_tests,
            success_count=success_count,
            failure_count=failure_count,
            error_types=error_types,
            recovery_success_rate=recovery_success_rate,
            average_recovery_time=avg_recovery_time
        )
    
    def test_concurrent_access(self, model_func: Callable, test_inputs: List[Any]) -> EdgeCaseTestResult:
        """Test model behavior under concurrent access."""
        n_threads = 10
        n_requests_per_thread = 5
        
        success_count = 0
        failure_count = 0
        error_types = {}
        recovery_times = []
        
        def worker_thread(thread_id):
            nonlocal success_count, failure_count, error_types, recovery_times
            
            for i in range(n_requests_per_thread):
                try:
                    start_time = time.time()
                    test_input = test_inputs[i % len(test_inputs)]
                    result = model_func(test_input)
                    recovery_time = time.time() - start_time
                    recovery_times.append(recovery_time)
                    success_count += 1
                except Exception as e:
                    error_type = type(e).__name__
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    failure_count += 1
        
        # Run concurrent threads
        threads = []
        for i in range(n_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_tests = n_threads * n_requests_per_thread
        recovery_success_rate = success_count / total_tests if total_tests > 0 else 0
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        return EdgeCaseTestResult(
            edge_case_name="concurrent_access",
            test_description="Testing model behavior under concurrent access patterns",
            n_tests=total_tests,
            success_count=success_count,
            failure_count=failure_count,
            error_types=error_types,
            recovery_success_rate=recovery_success_rate,
            average_recovery_time=avg_recovery_time
        )


class StressTester:
    """
    Performs stress testing for high-throughput scenarios.
    """
    
    def __init__(self):
        """Initialize the stress tester."""
        self.profiler = PerformanceProfiler()
    
    def run_stress_test(
        self, 
        model_func: Callable, 
        test_inputs: List[Any],
        duration: float = 60.0,
        target_throughput: float = 30.0
    ) -> StressTestResult:
        """
        Run comprehensive stress test.
        
        Args:
            model_func: Model function to test
            test_inputs: List of test inputs
            duration: Test duration in seconds
            target_throughput: Target throughput in images per minute
            
        Returns:
            StressTestResult with comprehensive stress test metrics
        """
        start_time = time.time()
        end_time = start_time + duration
        
        total_processed = 0
        successful_predictions = 0
        failed_predictions = 0
        
        throughput_measurements = []
        memory_measurements = []
        cpu_measurements = []
        
        measurement_interval = 5.0  # Measure every 5 seconds
        last_measurement_time = start_time
        last_processed_count = 0
        
        print(f"Starting stress test for {duration} seconds...")
        
        while time.time() < end_time:
            # Process batch of inputs
            batch_start = time.time()
            batch_size = min(10, len(test_inputs))
            
            for i in range(batch_size):
                try:
                    input_idx = total_processed % len(test_inputs)
                    result = model_func(test_inputs[input_idx])
                    successful_predictions += 1
                except Exception as e:
                    failed_predictions += 1
                
                total_processed += 1
                
                # Check if we should take measurements
                current_time = time.time()
                if current_time - last_measurement_time >= measurement_interval:
                    # Calculate throughput for this interval
                    interval_processed = total_processed - last_processed_count
                    interval_duration = current_time - last_measurement_time
                    interval_throughput = (interval_processed / interval_duration) * 60  # per minute
                    throughput_measurements.append(interval_throughput)
                    
                    # Measure system resources
                    memory_mb = self.profiler.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.profiler.process.cpu_percent()
                    
                    memory_measurements.append(memory_mb)
                    cpu_measurements.append(cpu_percent)
                    
                    # Update for next measurement
                    last_measurement_time = current_time
                    last_processed_count = total_processed
                
                # Break if time is up
                if time.time() >= end_time:
                    break
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        actual_duration = time.time() - start_time
        
        # Calculate final metrics
        average_throughput = (total_processed / actual_duration) * 60 if actual_duration > 0 else 0
        peak_throughput = max(throughput_measurements) if throughput_measurements else 0
        min_throughput = min(throughput_measurements) if throughput_measurements else 0
        
        peak_memory = max(memory_measurements) if memory_measurements else 0
        peak_cpu = max(cpu_measurements) if cpu_measurements else 0
        
        meets_throughput_target = average_throughput >= target_throughput
        
        # Calculate system stability score (based on throughput consistency)
        if len(throughput_measurements) > 1:
            throughput_cv = np.std(throughput_measurements) / np.mean(throughput_measurements)
            stability_score = max(0, 1 - throughput_cv)  # Lower CV = higher stability
        else:
            stability_score = 1.0 if meets_throughput_target else 0.0
        
        print(f"Stress test completed: {total_processed} images processed in {actual_duration:.1f}s")
        print(f"Average throughput: {average_throughput:.1f} images/minute")
        
        return StressTestResult(
            test_duration=actual_duration,
            total_images_processed=total_processed,
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            average_throughput=average_throughput,
            peak_throughput=peak_throughput,
            min_throughput=min_throughput,
            memory_usage_peak=peak_memory,
            cpu_usage_peak=peak_cpu,
            meets_throughput_target=meets_throughput_target,
            system_stability_score=stability_score
        )


class RobustnessTester:
    """
    Comprehensive robustness testing orchestrator.
    """
    
    def __init__(self, test_conditions: Optional[RobustnessTestConditions] = None):
        """
        Initialize the robustness tester.
        
        Args:
            test_conditions: Test conditions and parameters
        """
        self.test_conditions = test_conditions or RobustnessTestConditions()
        self.condition_simulator = ConditionSimulator()
        self.edge_case_tester = EdgeCaseTester()
        self.stress_tester = StressTester()
        self.profiler = PerformanceProfiler()
    
    def test_condition_robustness(
        self, 
        model_func: Callable, 
        test_inputs: List[Any],
        metadata: Optional[pd.DataFrame] = None
    ) -> List[ConditionTestResult]:
        """
        Test model performance across various conditions.
        
        Args:
            model_func: Model function to test
            test_inputs: List of test inputs
            metadata: Optional metadata for inputs
            
        Returns:
            List of ConditionTestResult objects
        """
        results = []
        
        # Test lighting conditions
        for lighting in self.test_conditions.lighting_conditions:
            print(f"Testing lighting condition: {lighting}")
            
            # Simulate condition and test
            modified_inputs = []
            for inp in test_inputs:
                modified_inp = self.condition_simulator.simulate_lighting_condition(inp, lighting)
                modified_inputs.append(modified_inp)
            
            result = self._test_single_condition(
                model_func, modified_inputs, f"lighting_{lighting}"
            )
            results.append(result)
        
        # Test device types
        for device in self.test_conditions.device_types:
            print(f"Testing device type: {device}")
            
            # Simulate device characteristics and test
            modified_inputs = []
            for inp in test_inputs:
                modified_inp = self.condition_simulator.simulate_device_characteristics(inp, device)
                modified_inputs.append(modified_inp)
            
            result = self._test_single_condition(
                model_func, modified_inputs, f"device_{device}"
            )
            results.append(result)
        
        # Test image qualities
        for quality in self.test_conditions.image_qualities:
            print(f"Testing image quality: {quality}")
            
            # Simulate quality and test
            modified_inputs = []
            for inp in test_inputs:
                modified_inp = self.condition_simulator.simulate_image_quality(inp, quality)
                modified_inputs.append(modified_inp)
            
            result = self._test_single_condition(
                model_func, modified_inputs, f"quality_{quality}"
            )
            results.append(result)
        
        return results
    
    def _test_single_condition(
        self, 
        model_func: Callable, 
        test_inputs: List[Any], 
        condition_name: str
    ) -> ConditionTestResult:
        """Test model under a single condition."""
        
        # Generate ground truth (dummy for testing)
        y_true = np.random.normal(12.0, 2.0, len(test_inputs))
        
        predictions = []
        inference_times = []
        errors = 0
        
        for i, inp in enumerate(test_inputs):
            try:
                start_time = time.time()
                pred = model_func(inp)
                inference_time = time.time() - start_time
                
                predictions.append(pred)
                inference_times.append(inference_time)
            except Exception as e:
                errors += 1
                predictions.append(np.nan)
                inference_times.append(np.nan)
        
        # Calculate metrics
        predictions = np.array(predictions)
        valid_mask = ~np.isnan(predictions)
        
        if np.sum(valid_mask) > 0:
            mae = np.mean(np.abs(y_true[valid_mask] - predictions[valid_mask]))
            rmse = np.sqrt(np.mean((y_true[valid_mask] - predictions[valid_mask]) ** 2))
        else:
            mae = float('inf')
            rmse = float('inf')
        
        inference_times = np.array(inference_times)
        valid_times = inference_times[~np.isnan(inference_times)]
        
        mean_time = np.mean(valid_times) if len(valid_times) > 0 else float('inf')
        std_time = np.std(valid_times) if len(valid_times) > 0 else 0
        
        error_rate = errors / len(test_inputs)
        success_rate = 1 - error_rate
        throughput = 60 / mean_time if mean_time > 0 else 0
        
        # Check targets
        meets_performance = mean_time <= 2.0  # 2 second target
        meets_accuracy = mae <= 0.8  # MAE target
        
        return ConditionTestResult(
            condition_name=condition_name.split('_')[0],
            condition_value=condition_name.split('_')[1] if '_' in condition_name else condition_name,
            n_samples=len(test_inputs),
            mae=mae,
            rmse=rmse,
            inference_time_mean=mean_time,
            inference_time_std=std_time,
            error_rate=error_rate,
            success_rate=success_rate,
            throughput=throughput,
            meets_performance_target=meets_performance,
            meets_accuracy_target=meets_accuracy
        )
    
    def run_comprehensive_robustness_test(
        self, 
        model_func: Callable, 
        test_inputs: List[Any],
        metadata: Optional[pd.DataFrame] = None
    ) -> RobustnessTestReport:
        """
        Run comprehensive robustness testing.
        
        Args:
            model_func: Model function to test
            test_inputs: List of test inputs
            metadata: Optional metadata for inputs
            
        Returns:
            RobustnessTestReport with comprehensive results
        """
        print("Starting comprehensive robustness testing...")
        
        # 1. Condition robustness testing
        print("Testing condition robustness...")
        condition_results = self.test_condition_robustness(model_func, test_inputs, metadata)
        
        # 2. Edge case testing
        print("Testing edge cases...")
        edge_case_results = [
            self.edge_case_tester.test_invalid_inputs(model_func),
            self.edge_case_tester.test_extreme_conditions(model_func, test_inputs),
            self.edge_case_tester.test_concurrent_access(model_func, test_inputs)
        ]
        
        # 3. Stress testing
        print("Running stress test...")
        stress_result = self.stress_tester.run_stress_test(
            model_func, test_inputs, 
            duration=self.test_conditions.stress_test_duration,
            target_throughput=self.test_conditions.min_throughput_target
        )
        
        # 4. Calculate overall scores
        robustness_score = self._calculate_robustness_score(condition_results, edge_case_results)
        consistency_score = self._calculate_consistency_score(condition_results)
        error_handling_score = self._calculate_error_handling_score(edge_case_results)
        
        # 5. Generate recommendations
        recommendations = self._generate_robustness_recommendations(
            condition_results, edge_case_results, stress_result
        )
        
        # 6. Check if all tests passed
        passed_all_tests = self._check_all_tests_passed(
            condition_results, edge_case_results, stress_result
        )
        
        return RobustnessTestReport(
            condition_test_results=condition_results,
            edge_case_test_results=edge_case_results,
            stress_test_result=stress_result,
            overall_robustness_score=robustness_score,
            performance_consistency_score=consistency_score,
            error_handling_score=error_handling_score,
            recommendations=recommendations,
            passed_all_tests=passed_all_tests
        )
    
    def _calculate_robustness_score(
        self, 
        condition_results: List[ConditionTestResult],
        edge_case_results: List[EdgeCaseTestResult]
    ) -> float:
        """Calculate overall robustness score."""
        
        # Condition robustness (50% weight)
        condition_scores = []
        for result in condition_results:
            accuracy_score = 1.0 if result.meets_accuracy_target else 0.5
            performance_score = 1.0 if result.meets_performance_target else 0.5
            success_score = result.success_rate
            condition_score = (accuracy_score + performance_score + success_score) / 3
            condition_scores.append(condition_score)
        
        avg_condition_score = np.mean(condition_scores) if condition_scores else 0
        
        # Edge case handling (50% weight)
        edge_case_scores = []
        for result in edge_case_results:
            edge_case_scores.append(result.recovery_success_rate)
        
        avg_edge_case_score = np.mean(edge_case_scores) if edge_case_scores else 0
        
        # Combined score
        overall_score = (avg_condition_score * 0.5) + (avg_edge_case_score * 0.5)
        
        return overall_score
    
    def _calculate_consistency_score(self, condition_results: List[ConditionTestResult]) -> float:
        """Calculate performance consistency score."""
        if not condition_results:
            return 0.0
        
        # Calculate coefficient of variation for MAE across conditions
        maes = [r.mae for r in condition_results if not np.isinf(r.mae)]
        if not maes:
            return 0.0
        
        cv_mae = np.std(maes) / np.mean(maes) if np.mean(maes) > 0 else 1.0
        
        # Calculate coefficient of variation for inference times
        times = [r.inference_time_mean for r in condition_results if not np.isinf(r.inference_time_mean)]
        if not times:
            return 0.0
        
        cv_time = np.std(times) / np.mean(times) if np.mean(times) > 0 else 1.0
        
        # Consistency score (lower CV = higher consistency)
        consistency_score = max(0, 1 - (cv_mae + cv_time) / 2)
        
        return consistency_score
    
    def _calculate_error_handling_score(self, edge_case_results: List[EdgeCaseTestResult]) -> float:
        """Calculate error handling score."""
        if not edge_case_results:
            return 0.0
        
        recovery_rates = [r.recovery_success_rate for r in edge_case_results]
        return np.mean(recovery_rates)
    
    def _generate_robustness_recommendations(
        self,
        condition_results: List[ConditionTestResult],
        edge_case_results: List[EdgeCaseTestResult],
        stress_result: StressTestResult
    ) -> List[str]:
        """Generate robustness improvement recommendations."""
        recommendations = []
        
        # Condition-specific recommendations
        failed_conditions = [r for r in condition_results if not (r.meets_accuracy_target and r.meets_performance_target)]
        if failed_conditions:
            condition_names = [r.condition_name for r in failed_conditions]
            recommendations.append(
                f"Performance issues detected under conditions: {', '.join(set(condition_names))}. "
                f"Consider robust preprocessing and domain adaptation techniques."
            )
        
        # Edge case recommendations
        poor_edge_cases = [r for r in edge_case_results if r.recovery_success_rate < 0.8]
        if poor_edge_cases:
            edge_case_names = [r.edge_case_name for r in poor_edge_cases]
            recommendations.append(
                f"Poor error handling for: {', '.join(edge_case_names)}. "
                f"Implement better input validation and error recovery mechanisms."
            )
        
        # Stress test recommendations
        if not stress_result.meets_throughput_target:
            recommendations.append(
                f"Throughput ({stress_result.average_throughput:.1f} img/min) below target "
                f"({self.test_conditions.min_throughput_target} img/min). "
                f"Consider performance optimization and parallel processing."
            )
        
        if stress_result.system_stability_score < 0.8:
            recommendations.append(
                "System stability issues detected during stress testing. "
                "Consider memory management improvements and resource optimization."
            )
        
        # Success message
        if not recommendations:
            recommendations.append(
                "All robustness tests passed! Model demonstrates excellent stability "
                "and performance across various conditions and edge cases."
            )
        
        return recommendations
    
    def _check_all_tests_passed(
        self,
        condition_results: List[ConditionTestResult],
        edge_case_results: List[EdgeCaseTestResult],
        stress_result: StressTestResult
    ) -> bool:
        """Check if all robustness tests passed."""
        
        # Check condition tests
        condition_passed = all(
            r.meets_accuracy_target and r.meets_performance_target and r.success_rate >= 0.95
            for r in condition_results
        )
        
        # Check edge case tests
        edge_case_passed = all(r.recovery_success_rate >= 0.8 for r in edge_case_results)
        
        # Check stress test
        stress_passed = (
            stress_result.meets_throughput_target and 
            stress_result.system_stability_score >= 0.8 and
            (stress_result.failed_predictions / stress_result.total_images_processed) <= self.test_conditions.error_tolerance
        )
        
        return condition_passed and edge_case_passed and stress_passed


def create_test_inputs(n_samples: int = 20) -> List[np.ndarray]:
    """Create dummy test inputs for robustness testing."""
    test_inputs = []
    
    for i in range(n_samples):
        # Create dummy image-like data
        if i % 4 == 0:
            # Feature vector
            test_inputs.append(np.random.randn(100))
        elif i % 4 == 1:
            # Small image
            test_inputs.append(np.random.rand(32, 32, 3))
        elif i % 4 == 2:
            # Large image
            test_inputs.append(np.random.rand(224, 224, 3))
        else:
            # 1D array
            test_inputs.append(np.random.randn(50))
    
    return test_inputs


def test_model_robustness(
    model_func: Callable,
    test_conditions: Optional[RobustnessTestConditions] = None
) -> RobustnessTestReport:
    """
    Test model robustness comprehensively.
    
    Args:
        model_func: Model function to test
        test_conditions: Test conditions and parameters
        
    Returns:
        RobustnessTestReport with comprehensive results
    """
    # Create test inputs
    test_inputs = create_test_inputs(20)
    
    # Run robustness testing
    tester = RobustnessTester(test_conditions)
    report = tester.run_comprehensive_robustness_test(model_func, test_inputs)
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    print("Robustness testing system loaded successfully")
    
    # Create dummy model for testing
    def dummy_model(inputs):
        """Dummy model that simulates hemoglobin prediction."""
        if inputs is None:
            raise ValueError("Input cannot be None")
        
        if isinstance(inputs, str):
            raise TypeError("Input must be numeric")
        
        if isinstance(inputs, np.ndarray):
            if inputs.size == 0:
                raise ValueError("Input array is empty")
            
            if np.any(np.isnan(inputs)):
                return 10.0  # Default value for NaN inputs
            
            if np.any(np.isinf(inputs)):
                return 15.0  # Default value for infinite inputs
            
            # Simulate prediction based on input statistics
            result = 12.0 + np.mean(inputs) * 0.1 + np.random.normal(0, 0.2)
            return max(0, min(20, result))  # Clamp to reasonable hemoglobin range
        
        return 12.0 + np.random.normal(0, 0.5)
    
    # Test robustness
    print("Running robustness testing...")
    try:
        # Create test conditions with shorter duration for demo
        test_conditions = RobustnessTestConditions(
            stress_test_duration=10.0,  # 10 seconds for demo
            min_throughput_target=60.0   # Higher target for testing
        )
        
        report = test_model_robustness(dummy_model, test_conditions)
        
        print(f"Robustness testing completed")
        print(f"All tests passed: {report.passed_all_tests}")
        print(f"Overall robustness score: {report.overall_robustness_score:.3f}")
        print(f"Performance consistency score: {report.performance_consistency_score:.3f}")
        print(f"Error handling score: {report.error_handling_score:.3f}")
        print(f"Stress test throughput: {report.stress_test_result.average_throughput:.1f} img/min")
        
        # Save report
        os.makedirs("robustness_reports", exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_dict = asdict(report)
        
        # Handle numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_for_json(d)
        
        report_dict = recursive_convert(report_dict)
        
        filename = f"robustness_test_report_{timestamp}.json"
        filepath = os.path.join("robustness_reports", filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Report saved to: {filepath}")
        
    except Exception as e:
        print(f"Robustness testing failed: {e}")
        import traceback
        traceback.print_exc()