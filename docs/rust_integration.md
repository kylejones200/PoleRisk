# Rust Integration Guide

This guide provides information about the Rust-optimized functions in the PoleRisk package, including installation, usage, and troubleshooting.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Performance Benefits](#performance-benefits)
- [Troubleshooting](#troubleshooting)
- [Extending with Rust](#extending-with-rust)
- [FAQ](#frequently-asked-questions)

## Overview

The polerisk package includes performance-critical functions implemented in Rust for better performance. These functions are automatically used when available, with fallbacks to pure Python implementations.

## Installation

### Prerequisites

- Python 3.8+
- Rust toolchain (install via [rustup](https://rustup.rs/))
- Maturin (Rust-Python bindings generator)

### Steps

1. Install the Rust toolchain:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Install Maturin:
   ```bash
   pip install maturin
   ```

3. Build and install the Rust extensions:
   ```bash
   cd polerisk_rs
   maturin develop --release
   ```

4. Verify the installation:
   ```python
   import polerisk_rs
   print("Rust extensions loaded successfully!")
   ```

## Performance Benefits

### Benchmark Results

| Function | Python (ms) | Rust (ms) | Speedup |
|----------|------------|-----------|---------|
| RMSE     | 145.2      | 12.8      | 11.3x   |
| MAE      | 98.7       | 10.2      | 9.7x    |
| Correlation | 210.5  | 15.3      | 13.8x   |

*Benchmarks performed on a dataset of 1,000,000 points on an Intel i7-10750H CPU.*

## Troubleshooting

### Common Issues

1. **Rust not installed**
   - Symptom: `error: can't find Rust compiler`
   - Solution: Install Rust using `rustup` as shown in the installation section.

2. **Maturin not found**
   - Symptom: `maturin: command not found`
   - Solution: Install Maturin with `pip install maturin`.

3. **Build errors**
   - Symptom: Various build errors during `maturin develop`
   - Solution:
     - Ensure you have the latest Rust toolchain: `rustup update`
     - Check that your Python version is supported
     - Check the [Maturin documentation](https://www.maturin.rs/) for platform-specific requirements

### Debugging

To enable debug output:

```bash
RUST_LOG=debug maturin develop
```

## Extending with Rust

### Adding New Functions

1. Add the function to `src/lib.rs`:
   ```rust
   use pyo3::prelude::*;
   use ndarray::Array1;

   #[pyfunction]
   pub fn my_new_function(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
       // Your implementation here
       Ok(0.0)
   }
   ```

2. Add it to the Python module in `lib.rs`:
   ```rust
   #[pymodule]
   fn polerisk_rs(_py: Python, m: &PyModule) -> PyResult<()> {
       m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
       Ok(())
   }
   ```

3. Rebuild the extension:
   ```bash
   maturin develop --release
   ```

## Frequently Asked Questions

### Q: Do I need Rust to use this package?
A: No, the package includes pure Python implementations as fallbacks. Rust is only required for optimal performance.

### Q: How do I know if Rust extensions are being used?
A: The package will print a warning if Rust extensions are not available. You can also check:

```python
from polerisk.analysis.statistics import RUST_AVAILABLE
print(f"Using Rust: {RUST_AVAILABLE}")
```

### Q: Can I use this on Windows?
A: Yes, but you'll need to install the Microsoft C++ Build Tools and ensure Rust is properly configured for your Python version.

### Q: How do I update the Rust extensions?
A: After updating the Rust code, simply rebuild:

```bash
cd polerisk_rs
maturin develop --release
```

## Performance Tips

1. **Batch processing**: Process data in chunks to reduce memory usage.
2. **Reuse arrays**: Reuse pre-allocated arrays when possible.
3. **Use appropriate data types**: Ensure your data is in the correct format (float64) before passing to Rust functions.

## Contributing

Contributions to the Rust implementation are welcome! Please ensure:

1. All tests pass (`pytest tests/`)
2. Benchmarks show performance improvements or maintain current performance
3. Documentation is updated accordingly

## License

The Rust components are licensed under the same terms as the main package (MIT). See the main LICENSE file for details.
