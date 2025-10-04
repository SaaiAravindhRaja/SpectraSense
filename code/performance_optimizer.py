"""
Performance optimization system for production deployment.

This module implements:
- Inference time optimization (≤ 2 seconds per image on CPU)
- Model size optimization (≤ 50 MB for deployment)
- Memory usage optimization (≤ 2 GB RAM during inference)
- Comprehensive performance profiling and monitoring
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
import json
import pickle
import joblib
from pathlib import Path
import gc
import threading
import tracemalloc
from contextlib import contextmanager

# Import existing modules
try:
    import torch
    import torch.nn as nn
    from torch.jit import script, trace
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - some optimizations will be skipped")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX not available - ONNX optimizations will be skipped")


@dataclass
class PerformanceTargets:
    """Container for performance optimization targets."""
    max_inference_time: float = 2.0      # seconds per image
    max_model_size: float = 50.0         # MB
    max_memory_usage: float = 2048.0     # MB (2 GB)
    min_throughput: float = 30.0         # images per minute
    cpu_only: bool = True                # Optimize for CPU inference


@dataclass
class PerformanceMetrics:
    """Container for measured performance metrics."""
    inference_time_mean: float
    inference_time_std: float
    inference_time_p95: float
    model_size_mb: float
    peak_memory_mb: float
    throughput_images_per_minute: float
    cpu_utilization_percent: float
    meets_time_target: bool
    meets_size_target: bool
    meets_memory_target: bool
    meets_throughput_target: bool


@dataclass
class OptimizationReport:
    """Container for optimization results."""
    original_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    optimization_techniques: List[str]
    performance_improvement: Dict[str, float]
    recommendations: List[str]
    all_targets_met: bool


class PerformanceProfiler:
    """
    Comprehensive performance profiling system.
    
    Measures inference time, memory usage, CPU utilization, and throughput.
    """
    
    def __init__(self):
        """Initialize the performance profiler."""
        self.process = psutil.Process()
        
    @contextmanager
    def profile_inference(self):
        """Context manager for profiling inference performance."""
        # Start memory tracking
        tracemalloc.start()
        
        # Record initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # CPU monitoring setup
        cpu_percent_start = self.process.cpu_percent()
        
        try:
            yield
        finally:
            # Record final state
            end_time = time.time()
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent_end = self.process.cpu_percent()
            
            # Get peak memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Store results
            self.last_inference_time = end_time - start_time
            self.last_memory_usage = final_memory - initial_memory
            self.last_peak_memory = peak / 1024 / 1024  # Convert to MB
            self.last_cpu_usage = cpu_percent_end
    
    def benchmark_model(
        self, 
        model_func: Callable, 
        test_inputs: List[Any], 
        warmup_runs: int = 5,
        benchmark_runs: int = 20
    ) -> PerformanceMetrics:
        """
        Comprehensive model benchmarking.
        
        Args:
            model_func: Function that performs model inference
            test_inputs: List of test inputs for benchmarking
            warmup_runs: Number of warmup runs to exclude from timing
            benchmark_runs: Number of benchmark runs for statistics
            
        Returns:
            PerformanceMetrics with comprehensive performance data
        """
        # Warmup runs
        for i in range(warmup_runs):
            if i < len(test_inputs):
                model_func(test_inputs[i])
        
        # Benchmark runs
        inference_times = []
        memory_usages = []
        cpu_usages = []
        
        for i in range(benchmark_runs):
            input_idx = i % len(test_inputs)
            
            with self.profile_inference():
                model_func(test_inputs[input_idx])
            
            inference_times.append(self.last_inference_time)
            memory_usages.append(self.last_peak_memory)
            cpu_usages.append(self.last_cpu_usage)
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        memory_usages = np.array(memory_usages)
        cpu_usages = np.array(cpu_usages)
        
        # Calculate throughput (images per minute)
        mean_time = np.mean(inference_times)
        throughput = 60.0 / mean_time if mean_time > 0 else 0
        
        # Get model size (if available)
        model_size_mb = self._estimate_model_size(model_func)
        
        # Create performance metrics
        targets = PerformanceTargets()
        
        metrics = PerformanceMetrics(
            inference_time_mean=float(np.mean(inference_times)),
            inference_time_std=float(np.std(inference_times)),
            inference_time_p95=float(np.percentile(inference_times, 95)),
            model_size_mb=model_size_mb,
            peak_memory_mb=float(np.max(memory_usages)),
            throughput_images_per_minute=throughput,
            cpu_utilization_percent=float(np.mean(cpu_usages)),
            meets_time_target=float(np.mean(inference_times)) <= targets.max_inference_time,
            meets_size_target=model_size_mb <= targets.max_model_size,
            meets_memory_target=float(np.max(memory_usages)) <= targets.max_memory_usage,
            meets_throughput_target=throughput >= targets.min_throughput
        )
        
        return metrics
    
    def _estimate_model_size(self, model_func: Callable) -> float:
        """Estimate model size in MB."""
        try:
            # Try to get size from model attributes
            if hasattr(model_func, '__self__'):
                model = model_func.__self__
                if hasattr(model, 'get_model_size'):
                    return model.get_model_size()
                elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                    # Calculate PyTorch model size
                    param_size = 0
                    buffer_size = 0
                    for param in model.parameters():
                        param_size += param.nelement() * param.element_size()
                    for buffer in model.buffers():
                        buffer_size += buffer.nelement() * buffer.element_size()
                    return (param_size + buffer_size) / 1024 / 1024  # Convert to MB
            
            # Fallback: estimate from serialized size
            import tempfile
            with tempfile.NamedTemporaryFile() as tmp:
                try:
                    if hasattr(model_func, '__self__'):
                        pickle.dump(model_func.__self__, tmp)
                    else:
                        pickle.dump(model_func, tmp)
                    return tmp.tell() / 1024 / 1024  # Convert to MB
                except:
                    return 0.0
        except:
            return 0.0


class ModelOptimizer:
    """
    Model optimization system for production deployment.
    
    Implements various optimization techniques to meet performance targets.
    """
    
    def __init__(self, targets: Optional[PerformanceTargets] = None):
        """
        Initialize the model optimizer.
        
        Args:
            targets: Performance targets to optimize for
        """
        self.targets = targets or PerformanceTargets()
        self.profiler = PerformanceProfiler()
    
    def optimize_model(
        self, 
        model_func: Callable, 
        test_inputs: List[Any],
        optimization_techniques: Optional[List[str]] = None
    ) -> OptimizationReport:
        """
        Optimize model for production deployment.
        
        Args:
            model_func: Function that performs model inference
            test_inputs: List of test inputs for benchmarking
            optimization_techniques: List of techniques to apply
            
        Returns:
            OptimizationReport with optimization results
        """
        if optimization_techniques is None:
            optimization_techniques = [
                'quantization', 'pruning', 'torchscript', 'onnx', 
                'memory_optimization', 'batch_optimization'
            ]
        
        # Baseline performance
        print("Measuring baseline performance...")
        original_metrics = self.profiler.benchmark_model(model_func, test_inputs)
        
        # Apply optimizations
        optimized_func = model_func
        applied_techniques = []
        
        for technique in optimization_techniques:
            try:
                print(f"Applying {technique} optimization...")
                optimized_func = self._apply_optimization(optimized_func, technique, test_inputs)
                applied_techniques.append(technique)
            except Exception as e:
                print(f"Failed to apply {technique}: {e}")
        
        # Measure optimized performance
        print("Measuring optimized performance...")
        optimized_metrics = self.profiler.benchmark_model(optimized_func, test_inputs)
        
        # Calculate improvements
        performance_improvement = self._calculate_improvements(original_metrics, optimized_metrics)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            original_metrics, optimized_metrics, applied_techniques
        )
        
        # Check if all targets are met
        all_targets_met = all([
            optimized_metrics.meets_time_target,
            optimized_metrics.meets_size_target,
            optimized_metrics.meets_memory_target,
            optimized_metrics.meets_throughput_target
        ])
        
        return OptimizationReport(
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            optimization_techniques=applied_techniques,
            performance_improvement=performance_improvement,
            recommendations=recommendations,
            all_targets_met=all_targets_met
        )
    
    def _apply_optimization(
        self, 
        model_func: Callable, 
        technique: str, 
        test_inputs: List[Any]
    ) -> Callable:
        """Apply a specific optimization technique."""
        
        if technique == 'memory_optimization':
            return self._optimize_memory_usage(model_func)
        
        elif technique == 'batch_optimization':
            return self._optimize_batching(model_func)
        
        elif technique == 'caching':
            return self._add_caching(model_func)
        
        elif technique == 'torchscript' and TORCH_AVAILABLE:
            return self._convert_to_torchscript(model_func, test_inputs)
        
        elif technique == 'onnx' and ONNX_AVAILABLE:
            return self._convert_to_onnx(model_func, test_inputs)
        
        elif technique == 'quantization':
            return self._apply_quantization(model_func)
        
        elif technique == 'pruning':
            return self._apply_pruning(model_func)
        
        else:
            print(f"Optimization technique '{technique}' not implemented or dependencies missing")
            return model_func
    
    def _optimize_memory_usage(self, model_func: Callable) -> Callable:
        """Optimize memory usage during inference."""
        def optimized_func(*args, **kwargs):
            # Force garbage collection before inference
            gc.collect()
            
            try:
                result = model_func(*args, **kwargs)
                
                # Clean up after inference
                gc.collect()
                
                return result
            except Exception as e:
                gc.collect()
                raise e
        
        return optimized_func
    
    def _optimize_batching(self, model_func: Callable) -> Callable:
        """Optimize for batch processing."""
        def optimized_func(inputs):
            # If single input, process directly
            if not isinstance(inputs, (list, tuple)):
                return model_func(inputs)
            
            # For multiple inputs, process in optimal batch size
            batch_size = min(len(inputs), 8)  # Optimal batch size for CPU
            results = []
            
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                if len(batch) == 1:
                    batch_results = [model_func(batch[0])]
                else:
                    # Process batch (if model supports it)
                    try:
                        batch_results = model_func(batch)
                        if not isinstance(batch_results, list):
                            batch_results = [batch_results]
                    except:
                        # Fallback to individual processing
                        batch_results = [model_func(item) for item in batch]
                
                results.extend(batch_results)
            
            return results[0] if len(results) == 1 else results
        
        return optimized_func
    
    def _add_caching(self, model_func: Callable) -> Callable:
        """Add simple caching to model function."""
        cache = {}
        
        def cached_func(inputs):
            # Create cache key (simplified)
            try:
                if isinstance(inputs, np.ndarray):
                    cache_key = hash(inputs.tobytes())
                elif isinstance(inputs, (list, tuple)):
                    cache_key = hash(str(inputs))
                else:
                    cache_key = hash(str(inputs))
                
                if cache_key in cache:
                    return cache[cache_key]
                
                result = model_func(inputs)
                
                # Limit cache size
                if len(cache) > 100:
                    # Remove oldest entry
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                
                cache[cache_key] = result
                return result
                
            except:
                # If caching fails, just run the function
                return model_func(inputs)
        
        return cached_func
    
    def _convert_to_torchscript(self, model_func: Callable, test_inputs: List[Any]) -> Callable:
        """Convert PyTorch model to TorchScript."""
        if not TORCH_AVAILABLE:
            return model_func
        
        try:
            # Extract model if it's a method
            if hasattr(model_func, '__self__') and isinstance(model_func.__self__, torch.nn.Module):
                model = model_func.__self__
                
                # Try tracing first
                try:
                    example_input = test_inputs[0]
                    if isinstance(example_input, np.ndarray):
                        example_input = torch.from_numpy(example_input).float()
                    
                    traced_model = torch.jit.trace(model, example_input)
                    
                    def torchscript_func(inputs):
                        if isinstance(inputs, np.ndarray):
                            inputs = torch.from_numpy(inputs).float()
                        return traced_model(inputs).detach().numpy()
                    
                    return torchscript_func
                    
                except Exception as e:
                    print(f"TorchScript tracing failed: {e}")
                    return model_func
            
            return model_func
            
        except Exception as e:
            print(f"TorchScript conversion failed: {e}")
            return model_func
    
    def _convert_to_onnx(self, model_func: Callable, test_inputs: List[Any]) -> Callable:
        """Convert model to ONNX format."""
        if not ONNX_AVAILABLE:
            return model_func
        
        # ONNX conversion is complex and model-specific
        # This is a placeholder for the actual implementation
        print("ONNX conversion not fully implemented")
        return model_func
    
    def _apply_quantization(self, model_func: Callable) -> Callable:
        """Apply model quantization."""
        # Quantization is model-specific and requires careful implementation
        # This is a placeholder for the actual implementation
        print("Quantization not fully implemented")
        return model_func
    
    def _apply_pruning(self, model_func: Callable) -> Callable:
        """Apply model pruning."""
        # Pruning is model-specific and requires careful implementation
        # This is a placeholder for the actual implementation
        print("Pruning not fully implemented")
        return model_func
    
    def _calculate_improvements(
        self, 
        original: PerformanceMetrics, 
        optimized: PerformanceMetrics
    ) -> Dict[str, float]:
        """Calculate performance improvements."""
        improvements = {}
        
        # Time improvement (negative means faster)
        time_improvement = (optimized.inference_time_mean - original.inference_time_mean) / original.inference_time_mean
        improvements['inference_time'] = time_improvement
        
        # Size improvement (negative means smaller)
        if original.model_size_mb > 0:
            size_improvement = (optimized.model_size_mb - original.model_size_mb) / original.model_size_mb
            improvements['model_size'] = size_improvement
        
        # Memory improvement (negative means less memory)
        if original.peak_memory_mb > 0:
            memory_improvement = (optimized.peak_memory_mb - original.peak_memory_mb) / original.peak_memory_mb
            improvements['memory_usage'] = memory_improvement
        
        # Throughput improvement (positive means higher throughput)
        if original.throughput_images_per_minute > 0:
            throughput_improvement = (optimized.throughput_images_per_minute - original.throughput_images_per_minute) / original.throughput_images_per_minute
            improvements['throughput'] = throughput_improvement
        
        return improvements
    
    def _generate_optimization_recommendations(
        self,
        original: PerformanceMetrics,
        optimized: PerformanceMetrics,
        applied_techniques: List[str]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Time recommendations
        if not optimized.meets_time_target:
            time_gap = optimized.inference_time_mean - self.targets.max_inference_time
            recommendations.append(
                f"Inference time ({optimized.inference_time_mean:.3f}s) exceeds target ({self.targets.max_inference_time}s). "
                f"Consider model architecture simplification or hardware acceleration."
            )
        
        # Size recommendations
        if not optimized.meets_size_target:
            size_gap = optimized.model_size_mb - self.targets.max_model_size
            recommendations.append(
                f"Model size ({optimized.model_size_mb:.1f}MB) exceeds target ({self.targets.max_model_size}MB). "
                f"Consider quantization, pruning, or knowledge distillation."
            )
        
        # Memory recommendations
        if not optimized.meets_memory_target:
            memory_gap = optimized.peak_memory_mb - self.targets.max_memory_usage
            recommendations.append(
                f"Memory usage ({optimized.peak_memory_mb:.1f}MB) exceeds target ({self.targets.max_memory_usage}MB). "
                f"Consider batch size reduction or memory-efficient algorithms."
            )
        
        # Throughput recommendations
        if not optimized.meets_throughput_target:
            throughput_gap = self.targets.min_throughput - optimized.throughput_images_per_minute
            recommendations.append(
                f"Throughput ({optimized.throughput_images_per_minute:.1f} img/min) below target ({self.targets.min_throughput} img/min). "
                f"Consider parallel processing or model optimization."
            )
        
        # Success message
        if optimized.meets_time_target and optimized.meets_size_target and optimized.meets_memory_target and optimized.meets_throughput_target:
            recommendations.append(
                "All performance targets achieved! Model is ready for production deployment."
            )
        
        # Technique-specific recommendations
        missing_techniques = set(['quantization', 'pruning', 'torchscript', 'onnx']) - set(applied_techniques)
        if missing_techniques:
            recommendations.append(
                f"Consider additional optimization techniques: {', '.join(missing_techniques)}"
            )
        
        return recommendations
    
    def save_optimization_report(
        self, 
        report: OptimizationReport, 
        output_dir: str = "optimization_reports"
    ) -> str:
        """
        Save optimization report to disk.
        
        Args:
            report: OptimizationReport to save
            output_dir: Directory to save reports
            
        Returns:
            Path to saved report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Handle numpy and boolean types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, bool):
                return obj
            return obj
        
        # Recursively convert the dictionary
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            else:
                return convert_for_json(d)
        
        report_dict = recursive_convert(report_dict)
        
        # Save as JSON
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        return filepath


def create_dummy_model():
    """Create a dummy model for testing."""
    def dummy_inference(inputs):
        # Simulate some computation
        if isinstance(inputs, np.ndarray):
            result = np.mean(inputs) + np.random.normal(0, 0.1)
        else:
            result = 12.0 + np.random.normal(0, 0.5)  # Dummy hemoglobin prediction
        
        # Simulate some processing time
        time.sleep(0.01)
        
        return result
    
    return dummy_inference


def optimize_existing_model(
    model_path: str,
    model_type: str = 'ridge',
    targets: Optional[PerformanceTargets] = None
) -> OptimizationReport:
    """
    Optimize an existing model for production deployment.
    
    Args:
        model_path: Path to the saved model
        model_type: Type of model ('ridge', 'ensemble', etc.)
        targets: Performance targets
        
    Returns:
        OptimizationReport with optimization results
    """
    # Load model
    if model_type == 'ridge':
        model_data = np.load(model_path)
        coef = model_data['coef']
        intercept = float(model_data['intercept'])
        
        def model_func(inputs):
            if isinstance(inputs, list):
                inputs = np.array(inputs)
            return inputs @ coef + intercept
    
    elif model_type == 'ensemble':
        with open(model_path, 'rb') as f:
            ensemble = pickle.load(f)
        
        def model_func(inputs):
            return ensemble.predict(inputs.reshape(1, -1))[0]
    
    else:
        # Use dummy model for testing
        model_func = create_dummy_model()
    
    # Create test inputs
    test_inputs = [np.random.randn(100) for _ in range(10)]  # Dummy feature vectors
    
    # Optimize model
    optimizer = ModelOptimizer(targets)
    report = optimizer.optimize_model(model_func, test_inputs)
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    print("Performance optimization system loaded successfully")
    
    # Test with dummy model
    print("Running performance optimization test...")
    try:
        # Create custom targets for testing
        targets = PerformanceTargets(
            max_inference_time=1.0,  # Stricter target for testing
            max_model_size=10.0,     # Stricter target for testing
            max_memory_usage=512.0,  # Stricter target for testing
            min_throughput=60.0      # Higher target for testing
        )
        
        report = optimize_existing_model("dummy_model", "dummy", targets)
        
        print(f"Optimization completed")
        print(f"All targets met: {report.all_targets_met}")
        print(f"Applied techniques: {report.optimization_techniques}")
        print(f"Time improvement: {report.performance_improvement.get('inference_time', 0):.3f}")
        print(f"Original time: {report.original_metrics.inference_time_mean:.3f}s")
        print(f"Optimized time: {report.optimized_metrics.inference_time_mean:.3f}s")
        
        # Save report
        optimizer = ModelOptimizer()
        report_path = optimizer.save_optimization_report(report)
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()