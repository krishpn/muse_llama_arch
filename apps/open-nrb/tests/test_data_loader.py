import importlib                                                                                                                                    
    from pathlib import Path                                                                                                                                           
    import tomllib                                                                                                                                                     
                                                                                                                                                                       
    def test_package_import():                                                                                                                                         
        """Dynamically verify that the current package imports cleanly."""
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            pkg_name = tomllib.load(f)["project"]["name"].replace("-", "_")
        
        pkg = importlib.import_module(pkg_name)
        assert pkg is not None
  
    def test_data_module_import():
        """Dynamically verify that the data subpackage imports cleanly."""
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            pkg_name = tomllib.load(f)["project"]["name"].replace("-", "_")
        
        data_mod = importlib.import_module(f"{pkg_name}.data")
        assert data_mod is not None
    
