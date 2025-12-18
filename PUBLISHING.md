# Publishing to PyPI

This guide explains how to publish the `polerisk` package to PyPI.

## Prerequisites

1. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

2. Install required tools:
```bash
pip install build twine
```

## Steps to Publish

### 1. Update Version Number

Edit the version in `pyproject.toml`:
```toml
[project]
version = "0.1.0"  # Update this
```

Also update in `polerisk/__init__.py`:
```python
__version__ = "0.1.0"  # Update this
```

### 2. Build the Package

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build
```

This creates files in the `dist/` directory:
- `polerisk-0.1.0.tar.gz` (source distribution)
- `polerisk-0.1.0-py3-none-any.whl` (wheel)

### 3. Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ polerisk
```

### 4. Upload to PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*
```

You'll be prompted for your PyPI username and password.

### 5. Verify Installation

```bash
pip install polerisk
```

## Using API Tokens (Recommended)

For better security, use API tokens instead of passwords:

1. Generate an API token on PyPI:
   - Go to Account Settings → API tokens
   - Create a token with appropriate scope

2. Create a `~/.pypirc` file:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YourTokenHere

[testpypi]
username = __token__
password = pypi-YourTestTokenHere
```

3. Set permissions:
```bash
chmod 600 ~/.pypirc
```

Now you can upload without entering credentials:
```bash
python -m twine upload dist/*
```

## Automating Releases with GitHub Actions

You can automate PyPI releases using GitHub Actions. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Add your PyPI token as a GitHub secret named `PYPI_API_TOKEN`.

## Quick Reference

```bash
# Build
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Check package before uploading
twine check dist/*
```

## Checklist Before Publishing

- [ ] Update version number in `pyproject.toml` and `__init__.py`
- [ ] Update `README.md` with any changes
- [ ] Run tests: `pytest`
- [ ] Check code formatting: `black polerisk tests`
- [ ] Build package: `python -m build`
- [ ] Check distribution: `twine check dist/*`
- [ ] Test on TestPyPI
- [ ] Create git tag: `git tag v0.1.0 && git push --tags`
- [ ] Upload to PyPI
- [ ] Create GitHub release

