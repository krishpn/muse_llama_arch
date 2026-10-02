import pytest


def test_package_import():
    """Verify that open_aether can be imported successfully."""
    import open_aether

    assert open_aether.__name__ == "open_aether"


def test_data_module_import():
    """Verify that open_aether.data subpackage imports clean."""
    import open_aether.data

    assert open_aether.data is not None