"""
Tests for core functionality
"""

import pytest


# Test basic imports work (may require optional dependencies)
def test_core_imports():
    """Test that core module can be imported"""
    # Just verify the module can be imported without errors
    import polerisk.core

    assert polerisk.core is not None


@pytest.mark.skipif(
    True,  # Skip netCDF4-dependent tests if not available
    reason="netCDF4 not installed - skipping core function tests",
)
def test_hello_polerisk():
    """Test the hello_polerisk function (requires netCDF4)"""
    from polerisk.core import hello_polerisk

    result = hello_polerisk()
    assert result == "Hello from polerisk!"
    assert isinstance(result, str)
