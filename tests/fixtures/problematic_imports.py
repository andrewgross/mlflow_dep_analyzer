"""
Fixture for creating problematic import scenarios to test edge cases.

This module creates various challenging import patterns including:
- Circular imports
- Deep relative imports
- Dynamic imports
- Missing dependencies
- Namespace packages
"""

from pathlib import Path


class ProblematicImportsFixture:
    """Creates problematic import scenarios for testing."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.fixture_root = self.base_path / "problematic_imports"

    def create_all_scenarios(self) -> Path:
        """Create all problematic import scenarios."""
        if self.fixture_root.exists():
            import shutil

            shutil.rmtree(self.fixture_root)

        self.fixture_root.mkdir(parents=True)

        # Create different problematic scenarios
        self._create_circular_imports()
        self._create_deep_relative_imports()
        self._create_dynamic_imports()
        self._create_missing_dependencies()
        self._create_namespace_packages()
        self._create_conditional_imports()
        self._create_star_imports()
        self._create_plugin_system()

        return self.fixture_root

    def _create_circular_imports(self):
        """Create circular import scenarios."""
        circular_dir = self.fixture_root / "circular"
        circular_dir.mkdir()
        (circular_dir / "__init__.py").touch()

        # Simple A -> B -> A circular import
        (circular_dir / "module_a.py").write_text('''
"""Module A that imports from B, creating circular dependency."""

import logging
import json
from typing import Dict, Any, Optional

# External dependencies
import pandas as pd
import numpy as np

# Circular import (B imports A)
from .module_b import process_data_b, DataProcessorB

# Standard imports
from ..utils.helper import utility_function
from ..config.settings import get_settings


class DataProcessorA:
    """Data processor that depends on B."""

    def __init__(self):
        self.processor_b = DataProcessorB()
        self.logger = logging.getLogger(__name__)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data using both A and B logic."""
        # Use our own logic
        cleaned_data = self._clean_data(data)

        # Use B's logic (circular dependency)
        processed_data = process_data_b(cleaned_data)

        return processed_data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data using A's specific logic."""
        return data.dropna()

    def get_config(self) -> Dict[str, Any]:
        """Get configuration settings."""
        return get_settings()


def process_data_a(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function that uses B's functionality."""
    processor = DataProcessorA()

    # Convert to DataFrame
    df = pd.DataFrame([data])
    processed = processor.process(df)

    # Convert back to dict
    return processed.iloc[0].to_dict()


# Module-level processing
if __name__ == "__main__":
    sample_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    processor = DataProcessorA()
    result = processor.process(sample_data)
    print(f"Processed data shape: {result.shape}")
''')

        (circular_dir / "module_b.py").write_text('''
"""Module B that imports from A, completing circular dependency."""

import logging
import json
from typing import Dict, Any, List, Optional
import datetime

# External dependencies
import pandas as pd
import numpy as np
import sklearn.preprocessing

# Circular import (A imports B) - importing only functions to avoid class circular reference
from .module_a import process_data_a

# Standard imports
from ..utils.validator import validate_data
from ..config.settings import get_database_config


class DataProcessorB:
    """Data processor that depends on A."""

    def __init__(self):
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.created_at = datetime.datetime.now()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data using both A and B logic."""
        # Use our own scaling logic
        scaled_data = self._scale_data(data)

        # Validate the data
        if not validate_data(scaled_data):
            raise ValueError("Data validation failed")

        return scaled_data

    def _scale_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale data using StandardScaler."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            data[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
        return data

    def process_with_a(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using A's functionality (circular call)."""
        # This creates the circular dependency
        return process_data_a(data_dict)


def process_data_b(data: pd.DataFrame) -> pd.DataFrame:
    """Function that can be called from A."""
    processor = DataProcessorB()
    return processor.process(data)


def complex_circular_operation(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Complex operation that uses both A and B in circular fashion."""
    processor_b = DataProcessorB()
    results = []

    for item in data:
        # Process with B first
        df = pd.DataFrame([item])
        processed_df = process_data_b(df)
        processed_item = processed_df.iloc[0].to_dict()

        # Then process with A (circular call)
        final_item = processor_b.process_with_a(processed_item)
        results.append(final_item)

    return results


# Module-level circular test
try:
    from .module_a import DataProcessorA
    _test_processor = DataProcessorA()
except ImportError as e:
    print(f"Circular import detected during module loading: {e}")
''')

        # More complex circular scenario with three modules
        (circular_dir / "module_x.py").write_text('''
"""Module X in a three-way circular import: X -> Y -> Z -> X."""

from typing import Dict, Any
import json
import pandas as pd

# Import from Y (Y imports from Z, Z imports from X)
from .module_y import process_with_y

# External dependencies
import requests
import numpy as np


class ProcessorX:
    """Processor X that uses Y and Z."""

    def __init__(self):
        self.name = "ProcessorX"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process using Y (which uses Z, which uses X)."""
        enhanced_data = {**data, "processed_by_x": True}
        return process_with_y(enhanced_data)

    def final_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Final processing step called by Z."""
        return {**data, "finalized_by_x": True, "timestamp": "2023-01-01"}


def process_with_x(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function that can be imported by other modules."""
    processor = ProcessorX()
    return processor.process(data)


def finalize_with_x(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function called by Z to complete the circle."""
    processor = ProcessorX()
    return processor.final_processing(data)
''')

        (circular_dir / "module_y.py").write_text('''
"""Module Y in a three-way circular import: X -> Y -> Z -> X."""

from typing import Dict, Any
import json
import pandas as pd

# Import from Z (Z imports from X, X imports from Y)
from .module_z import process_with_z

# External dependencies
import sklearn.metrics
import numpy as np


class ProcessorY:
    """Processor Y that uses Z and X."""

    def __init__(self):
        self.name = "ProcessorY"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process using Z (which uses X, which uses Y)."""
        enhanced_data = {**data, "processed_by_y": True}
        return process_with_z(enhanced_data)


def process_with_y(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function that can be imported by other modules."""
    processor = ProcessorY()
    return processor.process(data)
''')

        (circular_dir / "module_z.py").write_text('''
"""Module Z in a three-way circular import: X -> Y -> Z -> X."""

from typing import Dict, Any
import json
import pandas as pd

# Import from X (X imports from Y, Y imports from Z) - this completes the circle
from .module_x import finalize_with_x

# External dependencies
import torch
import numpy as np


class ProcessorZ:
    """Processor Z that uses X and Y."""

    def __init__(self):
        self.name = "ProcessorZ"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process using X (which uses Y, which uses Z) - completes circle."""
        enhanced_data = {**data, "processed_by_z": True}
        return finalize_with_x(enhanced_data)


def process_with_z(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function that can be imported by other modules."""
    processor = ProcessorZ()
    return processor.process(data)
''')

    def _create_deep_relative_imports(self):
        """Create deep relative import scenarios."""
        deep_dir = self.fixture_root / "deep_relative"
        deep_dir.mkdir()
        (deep_dir / "__init__.py").touch()

        # Create nested package structure: level1/level2/level3/level4/level5
        current_dir = deep_dir
        for level in range(1, 6):
            current_dir = current_dir / f"level{level}"
            current_dir.mkdir()
            (current_dir / "__init__.py").touch()

        # Level 5 module with complex relative imports
        (current_dir / "deep_module.py").write_text('''
"""Deep module with complex relative imports going up multiple levels."""

import json
import datetime
from typing import Dict, Any, List, Optional

# External dependencies
import pandas as pd
import numpy as np
import yaml
import requests

# Deep relative imports - go up multiple levels
from .....utils.helper import deep_utility  # 5 levels up
from ....level1.shared import shared_function  # 4 levels up
from ...level2.processor import level2_processor  # 3 levels up
from ..level4.validator import level4_validator  # 2 levels up
from .level5_helper import level5_helper  # same level

# Relative imports with module names
from .....config import global_config
from ....level1 import level1_constants
from ...level2.data import DataLoader as Level2DataLoader
from ..level4.models.base import BaseModel as Level4BaseModel


class DeepProcessor:
    """Processor with complex relative import dependencies."""

    def __init__(self):
        self.config = global_config.get_config()
        self.constants = level1_constants.CONSTANTS
        self.data_loader = Level2DataLoader()
        self.base_model = Level4BaseModel()

    def process_deep(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using functions from multiple levels."""

        # Use utility from 5 levels up
        enhanced_data = deep_utility(data)

        # Use shared function from 4 levels up
        validated_data = shared_function(enhanced_data)

        # Use processor from 3 levels up
        processed_data = level2_processor(validated_data)

        # Use validator from 2 levels up
        if not level4_validator(processed_data):
            raise ValueError("Level 4 validation failed")

        # Use helper from same level
        final_data = level5_helper(processed_data)

        return final_data

    def load_and_process(self, source: str) -> pd.DataFrame:
        """Load data and process using deep dependencies."""
        raw_data = self.data_loader.load(source)

        results = []
        for _, row in raw_data.iterrows():
            processed_row = self.process_deep(row.to_dict())
            results.append(processed_row)

        return pd.DataFrame(results)


# Module-level function that uses relative imports
def deep_processing_pipeline(data_source: str) -> Dict[str, Any]:
    """Pipeline function using deep relative imports."""
    processor = DeepProcessor()

    try:
        result_df = processor.load_and_process(data_source)

        return {
            "status": "success",
            "rows_processed": len(result_df),
            "columns": list(result_df.columns),
            "processing_timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "processing_timestamp": datetime.datetime.now().isoformat()
        }


# Try to import all dependencies at module level to test import resolution
try:
    from .....utils.metrics import MetricsCollector  # 5 levels up
    from ....level1.cache import CacheManager  # 4 levels up
    from ...level2.events import EventBus  # 3 levels up
    from ..level4.security import SecurityManager  # 2 levels up

    # Initialize module-level objects
    _metrics = MetricsCollector()
    _cache = CacheManager()
    _events = EventBus()
    _security = SecurityManager()

except ImportError as e:
    print(f"Failed to import deep dependencies: {e}")
    _metrics = None
    _cache = None
    _events = None
    _security = None
''')

        # Create level5_helper in the same directory
        (current_dir / "level5_helper.py").write_text('''
"""Helper module at level 5."""

import json
from typing import Dict, Any
import pandas as pd


def level5_helper(data: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function at level 5."""
    return {**data, "level5_processed": True}
''')

        # Create supporting modules at various levels
        level4_dir = deep_dir / "level1" / "level2" / "level3" / "level4"
        level4_dir.mkdir(exist_ok=True, parents=True)
        (level4_dir / "validator.py").write_text('''
"""Validator at level 4."""

def level4_validator(data):
    """Validate data at level 4."""
    return isinstance(data, dict) and len(data) > 0
''')

        level2_dir = deep_dir / "level1" / "level2"
        level2_dir.mkdir(exist_ok=True, parents=True)
        (level2_dir / "processor.py").write_text('''
"""Processor at level 2."""

def level2_processor(data):
    """Process data at level 2."""
    return {**data, "level2_processed": True}
''')

        level1_dir = deep_dir / "level1"
        level1_dir.mkdir(exist_ok=True, parents=True)
        (level1_dir / "shared.py").write_text('''
"""Shared functions at level 1."""

def shared_function(data):
    """Shared function at level 1."""
    return {**data, "level1_shared": True}
''')

    def _create_dynamic_imports(self):
        """Create dynamic import scenarios."""
        dynamic_dir = self.fixture_root / "dynamic"
        dynamic_dir.mkdir()
        (dynamic_dir / "__init__.py").touch()

        # Main module with dynamic imports
        (dynamic_dir / "dynamic_importer.py").write_text('''
"""Module that performs dynamic imports based on configuration and runtime conditions."""

import importlib
import importlib.util
import sys
import os
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path
import json
import inspect

# External dependencies
import pandas as pd
import numpy as np
import yaml


class DynamicImporter:
    """Handles dynamic imports based on configuration."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.loaded_modules = {}
        self.loaded_classes = {}
        self.loaded_functions = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load dynamic import configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)

        # Default configuration
        return {
            "plugins": [
                "dynamic.plugins.processor_plugin",
                "dynamic.plugins.validator_plugin",
                "dynamic.plugins.transformer_plugin"
            ],
            "conditional_imports": {
                "data_source": {
                    "csv": "dynamic.handlers.csv_handler",
                    "json": "dynamic.handlers.json_handler",
                    "parquet": "dynamic.handlers.parquet_handler",
                    "database": "dynamic.handlers.db_handler"
                },
                "model_type": {
                    "sklearn": "dynamic.models.sklearn_model",
                    "pytorch": "dynamic.models.pytorch_model",
                    "tensorflow": "dynamic.models.tensorflow_model"
                }
            }
        }

    def import_module_by_name(self, module_name: str) -> Any:
        """Dynamically import a module by name."""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]

        try:
            module = importlib.import_module(module_name)
            self.loaded_modules[module_name] = module
            return module
        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")
            return None

    def import_class_from_string(self, class_path: str) -> Optional[Type]:
        """Import a class from a string like 'module.submodule.ClassName'."""
        if class_path in self.loaded_classes:
            return self.loaded_classes[class_path]

        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self.loaded_classes[class_path] = cls
            return cls
        except (ImportError, AttributeError) as e:
            print(f"Failed to import class {class_path}: {e}")
            return None

    def import_function_from_string(self, func_path: str) -> Optional[Callable]:
        """Import a function from a string like 'module.submodule.function_name'."""
        if func_path in self.loaded_functions:
            return self.loaded_functions[func_path]

        try:
            module_path, func_name = func_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            self.loaded_functions[func_path] = func
            return func
        except (ImportError, AttributeError) as e:
            print(f"Failed to import function {func_path}: {e}")
            return None

    def load_plugins(self) -> Dict[str, Any]:
        """Load all configured plugins."""
        plugins = {}

        for plugin_path in self.config.get("plugins", []):
            plugin_module = self.import_module_by_name(plugin_path)
            if plugin_module:
                # Look for plugin classes or functions
                for name in dir(plugin_module):
                    obj = getattr(plugin_module, name)
                    if (inspect.isclass(obj) and name.endswith('Plugin')) or \
                       (inspect.isfunction(obj) and name.startswith('plugin_')):
                        plugins[f"{plugin_path}.{name}"] = obj

        return plugins

    def get_handler_for_type(self, handler_type: str, subtype: str) -> Optional[Any]:
        """Get a handler based on type and subtype."""
        conditional_imports = self.config.get("conditional_imports", {})

        if handler_type in conditional_imports:
            type_config = conditional_imports[handler_type]
            if subtype in type_config:
                handler_path = type_config[subtype]
                return self.import_module_by_name(handler_path)

        return None

    def import_from_file_path(self, file_path: str, module_name: str) -> Any:
        """Import a module from a specific file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        return None

    def discover_and_import_modules(self, directory: str, pattern: str = "*.py") -> Dict[str, Any]:
        """Discover and import modules from a directory."""
        discovered = {}
        directory_path = Path(directory)

        if not directory_path.exists():
            return discovered

        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and not file_path.name.startswith('__'):
                module_name = file_path.stem
                try:
                    module = self.import_from_file_path(str(file_path), module_name)
                    if module:
                        discovered[module_name] = module
                except Exception as e:
                    print(f"Failed to import {file_path}: {e}")

        return discovered


def create_processor_with_dynamic_imports(processor_type: str, config: Dict[str, Any]) -> Any:
    """Factory function that creates processors using dynamic imports."""

    importer = DynamicImporter()

    # Map processor types to module paths
    processor_map = {
        "pandas": "dynamic.processors.pandas_processor",
        "numpy": "dynamic.processors.numpy_processor",
        "sklearn": "dynamic.processors.sklearn_processor",
        "custom": config.get("custom_processor_path", "dynamic.processors.custom_processor")
    }

    if processor_type not in processor_map:
        raise ValueError(f"Unknown processor type: {processor_type}")

    processor_module = importer.import_module_by_name(processor_map[processor_type])
    if not processor_module:
        raise ImportError(f"Failed to import processor module for type: {processor_type}")

    # Look for processor class
    for name in dir(processor_module):
        obj = getattr(processor_module, name)
        if inspect.isclass(obj) and name.endswith('Processor'):
            return obj(**config)

    raise ValueError(f"No processor class found in {processor_map[processor_type]}")


# Module-level dynamic imports
try:
    # Import based on environment variable
    PROCESSOR_TYPE = os.environ.get('PROCESSOR_TYPE', 'pandas')

    if PROCESSOR_TYPE == 'pandas':
        from .processors import pandas_processor as active_processor
    elif PROCESSOR_TYPE == 'numpy':
        from .processors import numpy_processor as active_processor
    elif PROCESSOR_TYPE == 'sklearn':
        from .processors import sklearn_processor as active_processor
    else:
        # Try to import as module path
        importer = DynamicImporter()
        active_processor = importer.import_module_by_name(PROCESSOR_TYPE)

except ImportError as e:
    print(f"Failed to import active processor: {e}")
    active_processor = None

# Conditional imports based on available packages
try:
    import tensorflow as tf
    from .models import tensorflow_model
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tensorflow_model = None

try:
    import torch
    from .models import pytorch_model
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    pytorch_model = None

try:
    import xgboost
    from .models import xgboost_model
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgboost_model = None
''')

    def _create_missing_dependencies(self):
        """Create scenarios with missing dependencies."""
        missing_dir = self.fixture_root / "missing_deps"
        missing_dir.mkdir()
        (missing_dir / "__init__.py").touch()

        # Module with missing external dependencies
        (missing_dir / "missing_external.py").write_text('''
"""Module with missing external dependencies to test error handling."""

import os
import sys
import json
from typing import Dict, Any, List, Optional

# Standard library imports (should be available)
import logging
import datetime
import pathlib
import collections

# These external packages might not be installed
try:
    import nonexistent_package
    HAS_NONEXISTENT = True
except ImportError:
    HAS_NONEXISTENT = False
    nonexistent_package = None

try:
    import super_rare_package
    from super_rare_package.submodule import SpecialClass
    HAS_SUPER_RARE = True
except ImportError:
    HAS_SUPER_RARE = False
    super_rare_package = None
    SpecialClass = None

try:
    import fictional_ml_library as fml
    from fictional_ml_library.models import FictionalModel
    from fictional_ml_library.preprocessing import FictionalPreprocessor
    HAS_FICTIONAL_ML = True
except ImportError:
    HAS_FICTIONAL_ML = False
    fml = None
    FictionalModel = None
    FictionalPreprocessor = None

# Known packages that might not be installed
try:
    import tensorflow as tf
    import tensorflow.keras as keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None
    keras = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    optim = None

# Missing local dependencies
try:
    from .nonexistent_local_module import NonExistentClass
    from .missing_submodule.missing_class import MissingClass
    HAS_LOCAL_DEPS = True
except ImportError:
    HAS_LOCAL_DEPS = False
    NonExistentClass = None
    MissingClass = None


class ProcessorWithMissingDeps:
    """Processor that gracefully handles missing dependencies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_features = self._check_available_features()

    def _check_available_features(self) -> Dict[str, bool]:
        """Check which optional features are available."""
        return {
            "nonexistent_package": HAS_NONEXISTENT,
            "super_rare_package": HAS_SUPER_RARE,
            "fictional_ml": HAS_FICTIONAL_ML,
            "tensorflow": HAS_TENSORFLOW,
            "pytorch": HAS_PYTORCH,
            "local_deps": HAS_LOCAL_DEPS
        }

    def process_with_optional_deps(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using optional dependencies when available."""

        result = {"original_data": data, "processing_steps": []}

        # Use nonexistent package if available
        if HAS_NONEXISTENT and nonexistent_package:
            try:
                enhanced_data = nonexistent_package.enhance(data)
                result["enhanced_data"] = enhanced_data
                result["processing_steps"].append("nonexistent_package_enhancement")
            except Exception as e:
                self.logger.warning(f"Failed to use nonexistent_package: {e}")

        # Use fictional ML library if available
        if HAS_FICTIONAL_ML and FictionalModel:
            try:
                model = FictionalModel()
                predictions = model.predict([data])
                result["fictional_predictions"] = predictions
                result["processing_steps"].append("fictional_ml_prediction")
            except Exception as e:
                self.logger.warning(f"Failed to use fictional ML: {e}")

        # Use TensorFlow if available
        if HAS_TENSORFLOW and tf:
            try:
                # Create a simple tensor operation
                tensor_data = tf.constant(list(data.values())[:5])  # First 5 values
                processed_tensor = tf.reduce_mean(tensor_data)
                result["tensorflow_mean"] = float(processed_tensor.numpy())
                result["processing_steps"].append("tensorflow_processing")
            except Exception as e:
                self.logger.warning(f"Failed to use TensorFlow: {e}")

        # Use PyTorch if available
        if HAS_PYTORCH and torch:
            try:
                # Create a simple tensor operation
                tensor_data = torch.tensor(list(data.values())[:5], dtype=torch.float32)
                processed_tensor = torch.mean(tensor_data)
                result["pytorch_mean"] = float(processed_tensor.item())
                result["processing_steps"].append("pytorch_processing")
            except Exception as e:
                self.logger.warning(f"Failed to use PyTorch: {e}")

        # Use local dependencies if available
        if HAS_LOCAL_DEPS and NonExistentClass:
            try:
                processor = NonExistentClass()
                local_result = processor.process(data)
                result["local_processing"] = local_result
                result["processing_steps"].append("local_dependency_processing")
            except Exception as e:
                self.logger.warning(f"Failed to use local dependencies: {e}")

        return result

    def get_feature_report(self) -> Dict[str, Any]:
        """Get a report of available and missing features."""
        return {
            "available_features": self.available_features,
            "missing_features": [
                feature for feature, available in self.available_features.items()
                if not available
            ],
            "total_features": len(self.available_features),
            "available_count": sum(self.available_features.values())
        }


def process_with_fallbacks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data with fallbacks for missing dependencies."""

    processor = ProcessorWithMissingDeps()

    try:
        # Try to use the full processing pipeline
        result = processor.process_with_optional_deps(data)

        # Add feature report
        result["feature_report"] = processor.get_feature_report()

        return result

    except Exception as e:
        # Fallback to basic processing
        return {
            "original_data": data,
            "error": str(e),
            "fallback_processing": True,
            "processed_data": {**data, "basic_processing": True},
            "timestamp": datetime.datetime.now().isoformat()
        }


# Try to import and use missing dependencies at module level
try:
    from nonexistent_package.core import CoreProcessor
    module_processor = CoreProcessor()
except ImportError:
    module_processor = None

try:
    from .missing_local.processor import LocalProcessor
    local_processor = LocalProcessor()
except ImportError:
    local_processor = None

# Module-level constants that might use missing dependencies
if HAS_FICTIONAL_ML:
    DEFAULT_MODEL = FictionalModel()
    DEFAULT_PREPROCESSOR = FictionalPreprocessor()
else:
    DEFAULT_MODEL = None
    DEFAULT_PREPROCESSOR = None
''')

    def _create_namespace_packages(self):
        """Create namespace package scenarios."""
        namespace_dir = self.fixture_root / "namespace"
        namespace_dir.mkdir()
        # Note: No __init__.py for namespace packages

        # Create namespace package components
        ns1_dir = namespace_dir / "component1"
        ns1_dir.mkdir()
        (ns1_dir / "__init__.py").write_text('''
"""Component 1 of namespace package."""

from .processor import NamespaceProcessor1
from .utils import namespace_utility1

__all__ = ["NamespaceProcessor1", "namespace_utility1"]
''')

        (ns1_dir / "processor.py").write_text('''
"""Processor for namespace component 1."""

import pandas as pd
import numpy as np
from typing import Dict, Any


class NamespaceProcessor1:
    """Processor in namespace component 1."""

    def __init__(self):
        self.component_name = "component1"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data in component 1."""
        return {**data, "processed_by": self.component_name}
''')

        (ns1_dir / "utils.py").write_text('''
"""Utilities for namespace component 1."""

def namespace_utility1(data):
    """Utility function for component 1."""
    return f"Processed by component1: {data}"
''')

        # Component 2
        ns2_dir = namespace_dir / "component2"
        ns2_dir.mkdir()
        (ns2_dir / "__init__.py").write_text('''
"""Component 2 of namespace package."""

from .processor import NamespaceProcessor2
from .utils import namespace_utility2

__all__ = ["NamespaceProcessor2", "namespace_utility2"]
''')

        (ns2_dir / "processor.py").write_text('''
"""Processor for namespace component 2."""

import pandas as pd
import sklearn.preprocessing
from typing import Dict, Any

# Import from other namespace component
try:
    from ..component1.utils import namespace_utility1
    HAS_COMPONENT1 = True
except ImportError:
    HAS_COMPONENT1 = False
    namespace_utility1 = None


class NamespaceProcessor2:
    """Processor in namespace component 2."""

    def __init__(self):
        self.component_name = "component2"
        self.scaler = sklearn.preprocessing.StandardScaler()

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data in component 2."""
        result = {**data, "processed_by": self.component_name}

        # Use component1 if available
        if HAS_COMPONENT1 and namespace_utility1:
            result["component1_result"] = namespace_utility1(data)

        return result
''')

        (ns2_dir / "utils.py").write_text('''
"""Utilities for namespace component 2."""

def namespace_utility2(data):
    """Utility function for component 2."""
    return f"Processed by component2: {data}"
''')

    def _create_conditional_imports(self):
        """Create conditional import scenarios."""
        conditional_dir = self.fixture_root / "conditional"
        conditional_dir.mkdir()
        (conditional_dir / "__init__.py").touch()

        (conditional_dir / "conditional_processor.py").write_text('''
"""Module with conditional imports based on runtime conditions."""

import os
import sys
import platform
from typing import Dict, Any, Optional, Union

# Standard imports
import json
import logging
import datetime

# Conditional imports based on Python version
if sys.version_info >= (3, 8):
    from functools import cached_property
    HAS_CACHED_PROPERTY = True
else:
    HAS_CACHED_PROPERTY = False

    # Fallback implementation
    def cached_property(func):
        return property(func)

# Conditional imports based on platform
if platform.system() == "Windows":
    try:
        import winsound
        HAS_WINSOUND = True
    except ImportError:
        HAS_WINSOUND = False
        winsound = None
else:
    HAS_WINSOUND = False
    winsound = None

if platform.system() in ["Linux", "Darwin"]:
    try:
        import fcntl
        HAS_FCNTL = True
    except ImportError:
        HAS_FCNTL = False
        fcntl = None
else:
    HAS_FCNTL = False
    fcntl = None

# Conditional imports based on environment variables
FEATURE_FLAGS = {
    "USE_ADVANCED_PROCESSING": os.environ.get("USE_ADVANCED_PROCESSING", "false").lower() == "true",
    "ENABLE_CACHING": os.environ.get("ENABLE_CACHING", "true").lower() == "true",
    "DEBUG_MODE": os.environ.get("DEBUG_MODE", "false").lower() == "true"
}

if FEATURE_FLAGS["USE_ADVANCED_PROCESSING"]:
    try:
        import scipy.stats
        import scipy.optimize
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        scipy = None
else:
    HAS_SCIPY = False
    scipy = None

if FEATURE_FLAGS["ENABLE_CACHING"]:
    try:
        import redis
        import memcache
        HAS_CACHING = True
    except ImportError:
        HAS_CACHING = False
        redis = None
        memcache = None
else:
    HAS_CACHING = False
    redis = None
    memcache = None

# Conditional imports based on configuration file
CONFIG_FILE = os.environ.get("CONFIG_FILE", "config.json")
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        try:
            config = json.load(f)
            PROCESSOR_TYPE = config.get("processor_type", "basic")
            USE_EXTERNAL_API = config.get("use_external_api", False)
        except json.JSONDecodeError:
            PROCESSOR_TYPE = "basic"
            USE_EXTERNAL_API = False
else:
    PROCESSOR_TYPE = "basic"
    USE_EXTERNAL_API = False

if PROCESSOR_TYPE == "advanced" and HAS_SCIPY:
    from scipy import stats as scipy_stats
    from scipy import optimize as scipy_optimize
else:
    scipy_stats = None
    scipy_optimize = None

if USE_EXTERNAL_API:
    try:
        import requests
        import urllib3
        HAS_HTTP_LIBS = True
    except ImportError:
        HAS_HTTP_LIBS = False
        requests = None
        urllib3 = None
else:
    HAS_HTTP_LIBS = False
    requests = None
    urllib3 = None

# Runtime conditional imports
def import_based_on_data(data_type: str) -> Optional[Any]:
    """Import modules based on data type at runtime."""

    if data_type == "pandas" or data_type == "dataframe":
        try:
            import pandas as pd
            return pd
        except ImportError:
            return None
    elif data_type == "numpy" or data_type == "array":
        try:
            import numpy as np
            return np
        except ImportError:
            return None
    elif data_type == "image":
        try:
            from PIL import Image
            import cv2
            return {"PIL": Image, "cv2": cv2}
        except ImportError:
            return None
    elif data_type == "audio":
        try:
            import librosa
            import soundfile
            return {"librosa": librosa, "soundfile": soundfile}
        except ImportError:
            return None

    return None


class ConditionalProcessor:
    """Processor that uses conditional imports."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.capabilities = self._detect_capabilities()

    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available capabilities based on conditional imports."""
        return {
            "cached_property": HAS_CACHED_PROPERTY,
            "winsound": HAS_WINSOUND,
            "fcntl": HAS_FCNTL,
            "scipy": HAS_SCIPY,
            "caching": HAS_CACHING,
            "http_libs": HAS_HTTP_LIBS,
            "advanced_processing": FEATURE_FLAGS["USE_ADVANCED_PROCESSING"],
            "caching_enabled": FEATURE_FLAGS["ENABLE_CACHING"],
            "debug_mode": FEATURE_FLAGS["DEBUG_MODE"]
        }

    @cached_property if HAS_CACHED_PROPERTY else property
    def platform_info(self) -> Dict[str, str]:
        """Get platform information using cached property if available."""
        return {
            "system": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }

    def process_with_conditions(self, data: Dict[str, Any], data_type: str = "basic") -> Dict[str, Any]:
        """Process data using conditional features."""

        result = {
            "original_data": data,
            "capabilities": self.capabilities,
            "platform": self.platform_info,
            "processing_steps": []
        }

        # Use SciPy if available and enabled
        if HAS_SCIPY and scipy_stats:
            try:
                # Perform statistical analysis
                values = [v for v in data.values() if isinstance(v, (int, float))]
                if values:
                    result["scipy_stats"] = {
                        "mean": float(scipy_stats.describe(values).mean),
                        "variance": float(scipy_stats.describe(values).variance)
                    }
                    result["processing_steps"].append("scipy_statistical_analysis")
            except Exception as e:
                self.logger.warning(f"SciPy processing failed: {e}")

        # Use caching if available and enabled
        if HAS_CACHING and FEATURE_FLAGS["ENABLE_CACHING"]:
            try:
                # Simulate caching operation
                cache_key = f"processed_{hash(str(data))}"
                result["cache_key"] = cache_key
                result["processing_steps"].append("caching_enabled")
            except Exception as e:
                self.logger.warning(f"Caching failed: {e}")

        # Use HTTP libraries if available and enabled
        if HAS_HTTP_LIBS and USE_EXTERNAL_API:
            try:
                # Simulate API call
                result["external_api_available"] = True
                result["processing_steps"].append("external_api_ready")
            except Exception as e:
                self.logger.warning(f"HTTP setup failed: {e}")

        # Use platform-specific features
        if HAS_WINSOUND and platform.system() == "Windows":
            result["windows_sound_available"] = True
            result["processing_steps"].append("windows_platform_features")

        if HAS_FCNTL and platform.system() in ["Linux", "Darwin"]:
            result["unix_fcntl_available"] = True
            result["processing_steps"].append("unix_platform_features")

        # Import and use data-type specific modules
        data_module = import_based_on_data(data_type)
        if data_module:
            result["data_module_imported"] = data_type
            result["processing_steps"].append(f"{data_type}_module_loaded")

        return result


# Module-level conditional processing
if FEATURE_FLAGS["DEBUG_MODE"]:
    def debug_log(message: str):
        print(f"DEBUG: {message}")
else:
    def debug_log(message: str):
        pass

# Initialize module-level processor based on conditions
if PROCESSOR_TYPE == "advanced":
    _module_processor = ConditionalProcessor()
    debug_log("Advanced processor initialized")
else:
    _module_processor = None
    debug_log("Basic processor mode")
''')

    def _create_star_imports(self):
        """Create star import scenarios."""
        star_dir = self.fixture_root / "star_imports"
        star_dir.mkdir()
        (star_dir / "__init__.py").touch()

        # Base module with many exports
        (star_dir / "base_module.py").write_text('''
"""Base module with many exports for star import testing."""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union


# Functions
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data function."""
    return {**data, "processed": True}


def validate_data(data: Dict[str, Any]) -> bool:
    """Validate data function."""
    return isinstance(data, dict) and len(data) > 0


def transform_data(data: Dict[str, Any], transformation: str = "normalize") -> Dict[str, Any]:
    """Transform data function."""
    return {**data, "transformation": transformation}


def analyze_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data function."""
    return {"analysis": "completed", "input_keys": list(data.keys())}


# Classes
class DataProcessor:
    """Data processor class."""

    def __init__(self):
        self.name = "DataProcessor"

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return process_data(data)


class DataValidator:
    """Data validator class."""

    def __init__(self):
        self.name = "DataValidator"

    def validate(self, data: Dict[str, Any]) -> bool:
        return validate_data(data)


class DataTransformer:
    """Data transformer class."""

    def __init__(self):
        self.name = "DataTransformer"

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return transform_data(data)


class DataAnalyzer:
    """Data analyzer class."""

    def __init__(self):
        self.name = "DataAnalyzer"

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return analyze_data(data)


# Constants
DEFAULT_CONFIG = {
    "processor_type": "standard",
    "validation_enabled": True,
    "transformation_type": "normalize",
    "analysis_depth": "basic"
}

SUPPORTED_FORMATS = ["json", "csv", "parquet", "excel"]

PROCESSING_STAGES = ["validation", "transformation", "processing", "analysis"]

# Module-level variables
logger = logging.getLogger(__name__)
default_processor = DataProcessor()
default_validator = DataValidator()

# Complex objects
PROCESSING_PIPELINE = [
    DataValidator(),
    DataTransformer(),
    DataProcessor(),
    DataAnalyzer()
]

# Functions that use external dependencies
def pandas_operation(data: pd.DataFrame) -> pd.DataFrame:
    """Operation using pandas."""
    return data.fillna(0)


def numpy_operation(data: np.ndarray) -> np.ndarray:
    """Operation using numpy."""
    return np.mean(data, axis=0)


# Control what gets exported with star import
__all__ = [
    # Functions
    "process_data", "validate_data", "transform_data", "analyze_data",
    "pandas_operation", "numpy_operation",

    # Classes
    "DataProcessor", "DataValidator", "DataTransformer", "DataAnalyzer",

    # Constants
    "DEFAULT_CONFIG", "SUPPORTED_FORMATS", "PROCESSING_STAGES",

    # Variables
    "logger", "default_processor", "default_validator", "PROCESSING_PIPELINE"
]
''')

        # Module that uses star imports
        (star_dir / "star_importer.py").write_text('''
"""Module that uses star imports to test dependency resolution."""

import json
import sys
from typing import Dict, Any, List, Optional

# External dependencies
import pandas as pd
import numpy as np
import sklearn.preprocessing

# Star import from base module - imports everything in __all__
from .base_module import *

# Star import from external modules
from collections import *
from itertools import *

# Additional star imports from submodules
try:
    from .submodule.utilities import *
    HAS_UTILITIES = True
except ImportError:
    HAS_UTILITIES = False

try:
    from .submodule.advanced import *
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


class StarImportProcessor:
    """Processor that uses functions and classes from star imports."""

    def __init__(self):
        # Use imported classes (from base_module via star import)
        self.processor = DataProcessor()
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.analyzer = DataAnalyzer()

        # Use imported constants
        self.config = DEFAULT_CONFIG.copy()
        self.supported_formats = SUPPORTED_FORMATS
        self.pipeline_stages = PROCESSING_STAGES

        # Use imported pipeline
        self.pipeline = PROCESSING_PIPELINE

    def full_pipeline_process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data using the full pipeline from star imports."""

        # Use imported functions (from base_module via star import)
        if not validate_data(data):
            raise ValueError("Data validation failed")

        transformed = transform_data(data, "normalize")
        processed = process_data(transformed)
        analyzed = analyze_data(processed)

        # Use functions from collections (star import)
        result_counter = Counter(analyzed.keys())

        # Use functions from itertools (star import)
        combinations_list = list(combinations(analyzed.keys(), 2))

        return {
            **analyzed,
            "key_counter": dict(result_counter),
            "key_combinations": combinations_list,
            "pipeline_stages": self.pipeline_stages
        }

    def use_pandas_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use pandas operations from star import."""
        # Use imported pandas function
        return pandas_operation(df)

    def use_numpy_operations(self, arr: np.ndarray) -> np.ndarray:
        """Use numpy operations from star import."""
        # Use imported numpy function
        return numpy_operation(arr)

    def use_utilities_if_available(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Use utilities from star import if available."""
        result = data.copy()

        if HAS_UTILITIES:
            try:
                # These would be imported via star import from utilities
                result = utility_function(result)
                result = helper_function(result)
            except NameError:
                # Functions not available despite import success
                result["utilities_error"] = "Functions not found"

        if HAS_ADVANCED:
            try:
                # These would be imported via star import from advanced
                result = advanced_processor(result)
                result = complex_analyzer(result)
            except NameError:
                # Functions not available despite import success
                result["advanced_error"] = "Functions not found"

        return result


# Module-level usage of star-imported items
try:
    # Use imported logger
    logger.info("StarImportProcessor module loaded")

    # Use imported default objects
    _test_result = default_processor.process({"test": "data"})
    _validation_result = default_validator.validate({"test": "data"})

    # Use imported constants in module-level operations
    _module_config = {**DEFAULT_CONFIG, "module": "star_importer"}

except NameError as e:
    print(f"Failed to use star-imported items: {e}")


def demonstrate_star_import_usage():
    """Demonstrate usage of star-imported items."""

    processor = StarImportProcessor()

    sample_data = {
        "name": "test",
        "value": 42,
        "category": "sample"
    }

    try:
        result = processor.full_pipeline_process(sample_data)
        return {
            "success": True,
            "result": result,
            "star_imports_working": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "star_imports_working": False
        }


# Try to use star-imported items at module level
try:
    # Create objects using star-imported classes
    _module_analyzer = DataAnalyzer()
    _module_transformer = DataTransformer()

    # Use star-imported functions
    _sample_data = {"module_test": True}
    _validated = validate_data(_sample_data)
    _processed = process_data(_sample_data)

    STAR_IMPORTS_SUCCESSFUL = True

except NameError:
    STAR_IMPORTS_SUCCESSFUL = False
''')

        # Create submodules for additional star imports
        submodule_dir = star_dir / "submodule"
        submodule_dir.mkdir()
        (submodule_dir / "__init__.py").touch()

        (submodule_dir / "utilities.py").write_text('''
"""Utilities module for star import testing."""

def utility_function(data):
    """Utility function."""
    return {**data, "utility_applied": True}


def helper_function(data):
    """Helper function."""
    return {**data, "helper_applied": True}


__all__ = ["utility_function", "helper_function"]
''')

        (submodule_dir / "advanced.py").write_text('''
"""Advanced module for star import testing."""

def advanced_processor(data):
    """Advanced processor function."""
    return {**data, "advanced_processing": True}


def complex_analyzer(data):
    """Complex analyzer function."""
    return {**data, "complex_analysis": True}


__all__ = ["advanced_processor", "complex_analyzer"]
''')

    def _create_plugin_system(self):
        """Create a plugin system with dynamic loading."""
        plugin_dir = self.fixture_root / "plugin_system"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").touch()

        # Plugin manager
        (plugin_dir / "manager.py").write_text('''
"""Plugin manager for dynamic plugin loading."""

import importlib
import importlib.util
import sys
import os
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Protocol
import json
import logging

# External dependencies
import yaml
import pandas as pd


class PluginInterface(Protocol):
    """Protocol defining the plugin interface."""

    name: str
    version: str

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        ...

    def process(self, data: Any) -> Any:
        """Process data."""
        ...

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        ...


class PluginManager:
    """Manages dynamic loading and execution of plugins."""

    def __init__(self, plugin_dir: str, config_file: Optional[str] = None):
        self.plugin_dir = Path(plugin_dir)
        self.config = self._load_config(config_file)
        self.plugins: Dict[str, PluginInterface] = {}
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load plugin configuration."""
        if config_file and os.path.exists(config_file):
            with open(config_file) as f:
                if config_file.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f) or {}

        return {
            "enabled_plugins": ["*"],  # Enable all plugins
            "plugin_config": {},
            "load_order": []
        }

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugin directory."""
        plugins = []

        if not self.plugin_dir.exists():
            return plugins

        # Look for Python files that might be plugins
        for plugin_file in self.plugin_dir.glob("*_plugin.py"):
            plugin_name = plugin_file.stem
            plugins.append(plugin_name)

        # Look for plugin packages
        for plugin_dir in self.plugin_dir.iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "__init__.py").exists():
                plugins.append(plugin_dir.name)

        return plugins

    def load_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Load a single plugin."""
        try:
            # Try loading as a file first
            plugin_file = self.plugin_dir / f"{plugin_name}.py"
            if plugin_file.exists():
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[plugin_name] = module
                    spec.loader.exec_module(module)
            else:
                # Try loading as a package
                plugin_package = f"{self.plugin_dir.name}.{plugin_name}"
                module = importlib.import_module(plugin_package)

            # Look for plugin class
            plugin_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and
                    hasattr(obj, 'name') and
                    hasattr(obj, 'process') and
                    not name.startswith('_')):
                    plugin_class = obj
                    break

            if plugin_class:
                # Get plugin-specific config
                plugin_config = self.config.get("plugin_config", {}).get(plugin_name, {})

                # Instantiate and initialize plugin
                plugin_instance = plugin_class()
                if hasattr(plugin_instance, 'initialize'):
                    plugin_instance.initialize(plugin_config)

                self.logger.info(f"Loaded plugin: {plugin_name}")
                return plugin_instance
            else:
                self.logger.warning(f"No valid plugin class found in {plugin_name}")

        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")

        return None

    def load_all_plugins(self):
        """Load all discovered plugins."""
        discovered = self.discover_plugins()
        enabled = self.config.get("enabled_plugins", ["*"])

        # Determine which plugins to load
        if "*" in enabled:
            plugins_to_load = discovered
        else:
            plugins_to_load = [p for p in discovered if p in enabled]

        # Apply load order if specified
        load_order = self.config.get("load_order", [])
        ordered_plugins = []

        # Add plugins in specified order first
        for plugin_name in load_order:
            if plugin_name in plugins_to_load:
                ordered_plugins.append(plugin_name)

        # Add remaining plugins
        for plugin_name in plugins_to_load:
            if plugin_name not in ordered_plugins:
                ordered_plugins.append(plugin_name)

        # Load plugins in order
        for plugin_name in ordered_plugins:
            plugin = self.load_plugin(plugin_name)
            if plugin:
                self.plugins[plugin_name] = plugin

        self.logger.info(f"Loaded {len(self.plugins)} plugins")

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin by name."""
        return self.plugins.get(plugin_name)

    def process_with_plugin(self, plugin_name: str, data: Any) -> Any:
        """Process data with a specific plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.process(data)
        else:
            raise ValueError(f"Plugin not found: {plugin_name}")

    def process_with_all_plugins(self, data: Any) -> Dict[str, Any]:
        """Process data with all loaded plugins."""
        results = {}

        for plugin_name, plugin in self.plugins.items():
            try:
                result = plugin.process(data)
                results[plugin_name] = result
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} failed: {e}")
                results[plugin_name] = {"error": str(e)}

        return results

    def cleanup_all_plugins(self):
        """Clean up all loaded plugins."""
        for plugin_name, plugin in self.plugins.items():
            try:
                if hasattr(plugin, 'cleanup'):
                    plugin.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup plugin {plugin_name}: {e}")


# Global plugin manager instance
_plugin_manager = None


def get_plugin_manager(plugin_dir: str, config_file: Optional[str] = None) -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager(plugin_dir, config_file)
        _plugin_manager.load_all_plugins()
    return _plugin_manager
''')

        # Sample plugins
        plugins_subdir = plugin_dir / "plugins"
        plugins_subdir.mkdir()
        (plugins_subdir / "__init__.py").touch()

        (plugins_subdir / "data_processor_plugin.py").write_text('''
"""Data processing plugin."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging


class DataProcessorPlugin:
    """Plugin for data processing operations."""

    name = "DataProcessor"
    version = "1.0.0"

    def __init__(self):
        self.logger = logging.getLogger(f"plugin.{self.name}")
        self.config = {}

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        self.config = config
        self.logger.info(f"Initialized {self.name} plugin")

    def process(self, data: Any) -> Any:
        """Process data."""
        if isinstance(data, dict):
            return self._process_dict(data)
        elif isinstance(data, pd.DataFrame):
            return self._process_dataframe(data)
        elif isinstance(data, list):
            return self._process_list(data)
        else:
            return {"error": f"Unsupported data type: {type(data)}"}

    def _process_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process dictionary data."""
        return {
            **data,
            "processed_by": self.name,
            "processing_timestamp": "2023-01-01T00:00:00Z",
            "data_keys_count": len(data.keys())
        }

    def _process_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame data."""
        result = data.copy()
        result[f"processed_by_{self.name}"] = True
        return result

    def _process_list(self, data: list) -> list:
        """Process list data."""
        return data + [f"processed_by_{self.name}"]

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.logger.info(f"Cleaning up {self.name} plugin")
''')

        (plugins_subdir / "validator_plugin.py").write_text('''
"""Data validation plugin."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import jsonschema


class ValidatorPlugin:
    """Plugin for data validation."""

    name = "Validator"
    version = "1.0.0"

    def __init__(self):
        self.logger = logging.getLogger(f"plugin.{self.name}")
        self.config = {}
        self.schema = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        self.config = config
        self.schema = config.get("validation_schema")
        self.logger.info(f"Initialized {self.name} plugin")

    def process(self, data: Any) -> Any:
        """Validate data."""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "validated_by": self.name
        }

        try:
            if isinstance(data, dict):
                self._validate_dict(data, validation_result)
            elif isinstance(data, pd.DataFrame):
                self._validate_dataframe(data, validation_result)
            elif isinstance(data, list):
                self._validate_list(data, validation_result)
            else:
                validation_result["warnings"].append(f"Unknown data type: {type(data)}")

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(str(e))

        return validation_result

    def _validate_dict(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate dictionary data."""
        if not data:
            result["warnings"].append("Empty dictionary")

        # Check for required keys if schema provided
        if self.schema and "required" in self.schema:
            missing_keys = set(self.schema["required"]) - set(data.keys())
            if missing_keys:
                result["is_valid"] = False
                result["errors"].append(f"Missing required keys: {missing_keys}")

        # Validate against JSON schema if provided
        if self.schema:
            try:
                jsonschema.validate(data, self.schema)
            except jsonschema.ValidationError as e:
                result["is_valid"] = False
                result["errors"].append(f"Schema validation failed: {e.message}")

    def _validate_dataframe(self, data: pd.DataFrame, result: Dict[str, Any]) -> None:
        """Validate DataFrame data."""
        if data.empty:
            result["warnings"].append("Empty DataFrame")

        # Check for null values
        null_counts = data.isnull().sum()
        if null_counts.any():
            result["warnings"].append(f"Null values found: {null_counts.to_dict()}")

        # Check data types
        result["data_types"] = data.dtypes.to_dict()

    def _validate_list(self, data: List[Any], result: Dict[str, Any]) -> None:
        """Validate list data."""
        if not data:
            result["warnings"].append("Empty list")

        # Check for consistent types
        if data:
            first_type = type(data[0])
            if not all(isinstance(item, first_type) for item in data):
                result["warnings"].append("Inconsistent data types in list")

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.logger.info(f"Cleaning up {self.name} plugin")
''')


def create_problematic_imports_fixture(base_path: Path) -> Path:
    """Create all problematic import scenarios."""
    fixture = ProblematicImportsFixture(base_path)
    return fixture.create_all_scenarios()
