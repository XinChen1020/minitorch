import pytest


# Register custom markers with descriptions
def pytest_configure(config):
    for module in range(5):  # Modules 0 through 4
        for task in range(1, 5):  # Tasks 1 through 4
            config.addinivalue_line(
                "markers", f"task{module}_{task}: Tests for Module {module}, Task {task}"
            )
