"""
Performance benchmarking of Rust-optimized statistical functions vs Python implementations.
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
from polerisk.analysis import (
    calculate_rmse,
    calculate_correlation,
    calculate_mae,
    calculate_bias,
    calculate_ubrmse,
    RUST_AVAILABLE
)

# Python implementations for benchmarking
def python_rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))

def python_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

def python_mae(x, y):
    return np.mean(np.abs(x - y))

def python_bias(x, y):
    return np.mean(y - x)

def python_ubrmse(x, y):
    bias = python_bias(x, y)
    return np.sqrt(np.mean((y - x - bias) ** 2))

# Benchmarking function
def benchmark_functions(sizes=[100, 1000, 10000, 100000, 1000000], num_runs=100):
    """
    Benchmark Rust vs Python implementations across different array sizes.
    
    Args:
        sizes: List of array sizes to test
        num_runs: Number of runs per test for averaging
        
    Returns:
        dict: Dictionary containing timing results
    """
    functions = [
        ('RMSE', calculate_rmse, python_rmse),
        ('Correlation', calculate_correlation, python_correlation),
        ('MAE', calculate_mae, python_mae),
        ('Bias', calculate_bias, python_bias),
        ('ubRMSE', calculate_ubrmse, python_ubrmse)
    ]
    
    results = {}
    
    for size in sizes:
        logger.debug(f"\nBenchmarking array size: {size:,}")
        logger.debug("-" * 40)
        
        # Generate test data
        np.random.seed(42)
        x = np.random.rand(size) * 10
        y = x + np.random.normal(0, 1, size)
        
        size_results = {}
        
        for name, rust_func, py_func in functions:
            # Benchmark Rust
            rust_time = timeit.timeit(
                lambda: rust_func(x, y),
                number=num_runs
            ) / num_runs * 1000  # Convert to milliseconds
            
            # Benchmark Python
            py_time = timeit.timeit(
                lambda: py_func(x, y),
                number=num_runs
            ) / num_runs * 1000  # Convert to milliseconds
            
            speedup = py_time / rust_time if rust_time > 0 else float('inf')
            
            logger.debug(f"{name}:")
            logger.debug(f"  Rust: {rust_time:.6f} ms/op")
            logger.debug(f"  Python: {py_time:.6f} ms/op")
            logger.debug(f"  Speedup: {speedup:.2f}x")
            
            size_results[name] = {
                'rust_time': rust_time,
                'python_time': py_time,
                'speedup': speedup
            }
            
        results[size] = size_results
    
    return results

def plot_results(results, output_file='performance_comparison.png'):
    """Plot the benchmarking results."""
    sizes = sorted(results.keys())
    functions = list(next(iter(results.values())).keys())
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot execution times
    for func in functions:
        rust_times = [results[size][func]['rust_time'] for size in sizes]
        py_times = [results[size][func]['python_time'] for size in sizes]
        
        ax1.plot(sizes, rust_times, 'o-', label=f'Rust {func}')
        ax1.plot(sizes, py_times, 'x--', label=f'Python {func}')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Array Size')
    ax1.set_ylabel('Time per operation (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    # Plot speedup factors
    for func in functions:
        speedups = [results[size][func]['speedup'] for size in sizes]
        ax2.plot(sizes, speedups, 'o-', label=func)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Array Size')
    ax2.set_ylabel('Speedup Factor (Python/Rust)')
    ax2.set_title('Speedup of Rust vs Python')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.debug(f"\nPerformance plot saved to {output_file}")

def main():
    logger.debug("Performance Benchmark: Rust vs Python Statistical Functions")
    logger.debug("=" * 60)
    logger.debug(f"Using Rust optimizations: {RUST_AVAILABLE}")
    
    # Run benchmarks
    results = benchmark_functions()
    
    # Generate and save plots
    plot_results(results)
    
    # Print summary
    logger.debug("\n" + "=" * 60)
    logger.debug("Performance Benchmark Summary")
    logger.debug("=" * 60)
    
    for size, funcs in results.items():
        logger.debug(f"\nArray size: {size:,}")
        logger.debug("-" * 40)
        for func, times in funcs.items():
            logger.info(f"{func:>10} - Rust: {times['rust_time']:.6f} ms, "
                  f"Python: {times['python_time']:.6f} ms, "
                  f"Speedup: {times['speedup']:.2f}x")

if __name__ == "__main__":
    main()
