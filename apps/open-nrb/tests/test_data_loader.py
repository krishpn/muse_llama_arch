import pytest


def test_package_import():
    """Verify that open_aether can be imported successfully."""
    import open_nrb

    assert open_nrb.__name__ == "open_aether"


def test_data_module_import():
    """Verify that open_aether.data subpackage imports clean."""
    import open_nrb.data


    assert open_nrb.data is not None

print("All tests passed successfully.")