"""Tests to validate the testing infrastructure setup."""

import sys
import pytest
from pathlib import Path
from unittest.mock import Mock

# Test markers work
pytestmark = pytest.mark.unit


def test_python_version():
    """Test that we're running a compatible Python version."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"


def test_pytest_working():
    """Basic test to verify pytest is working."""
    assert True


def test_fixtures_available(temp_dir, mock_config, mock_logger):
    """Test that shared fixtures are available and working."""
    # Test temp_dir fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test mock_config fixture
    assert isinstance(mock_config, dict)
    assert "app_name" in mock_config
    assert mock_config["app_name"] == "test_app"
    
    # Test mock_logger fixture
    assert hasattr(mock_logger, "info")
    assert hasattr(mock_logger, "error")
    mock_logger.info("test message")
    mock_logger.info.assert_called_once_with("test message")


def test_temp_file_fixture(temp_file):
    """Test that temp_file fixture creates a temporary file."""
    assert temp_file.exists()
    assert temp_file.is_file()
    
    # Write to the file
    temp_file.write_text("test content")
    content = temp_file.read_text()
    assert content == "test content"


def test_mock_functionality(mock_config):
    """Test that mocking functionality works correctly."""
    # Test that we can modify mock objects
    mock_config["test_key"] = "test_value"
    assert mock_config["test_key"] == "test_value"


def test_sample_data_fixtures(sample_image_data, sample_audio_data, sample_video_metadata):
    """Test that sample data fixtures provide expected data types."""
    # Test image data
    assert isinstance(sample_image_data, bytes)
    assert len(sample_image_data) > 0
    assert sample_image_data.startswith(b'\x89PNG')  # PNG header
    
    # Test audio data
    assert isinstance(sample_audio_data, bytes)
    assert len(sample_audio_data) > 0
    assert sample_audio_data.startswith(b'RIFF')  # WAV header
    
    # Test video metadata
    assert isinstance(sample_video_metadata, dict)
    assert "width" in sample_video_metadata
    assert "height" in sample_video_metadata
    assert sample_video_metadata["width"] == 1920
    assert sample_video_metadata["height"] == 1080


def test_environment_variables_fixture(environment_variables):
    """Test that environment variables fixture works."""
    import os
    
    assert "TEST_MODE" in os.environ
    assert os.environ["TEST_MODE"] == "true"
    assert "DEBUG" in os.environ
    assert os.environ["DEBUG"] == "1"


def test_project_structure(project_root):
    """Test that we can find the project structure."""
    assert project_root.exists()
    assert project_root.is_dir()
    
    # Check for key project files
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml should exist"
    
    launch_file_path = project_root / "LaunchFile"
    assert launch_file_path.exists(), "LaunchFile directory should exist"


def test_mock_objects(mock_api_client, mock_torch_model, mock_gradio_interface):
    """Test that mock objects are properly configured."""
    # Test API client mock
    response = mock_api_client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    # Test torch model mock
    assert hasattr(mock_torch_model, "eval")
    assert hasattr(mock_torch_model, "forward")
    model_in_eval = mock_torch_model.eval()
    assert model_in_eval is mock_torch_model
    
    # Test gradio interface mock
    assert hasattr(mock_gradio_interface, "launch")
    assert hasattr(mock_gradio_interface, "predict")
    prediction = mock_gradio_interface.predict("test_input")
    assert prediction == ["test_output"]


def test_marker_functionality():
    """Test that pytest markers are working."""
    # This test should be marked with 'unit' marker
    # We can't easily test this from within the test, but we can verify
    # that the marker system is configured by checking if the test runs
    assert hasattr(pytest.mark, "unit")
    assert hasattr(pytest.mark, "integration")
    assert hasattr(pytest.mark, "slow")


@pytest.mark.slow
def test_slow_marker():
    """Test marked as slow (for testing marker functionality)."""
    assert True


class TestClassStructure:
    """Test class to verify class-based test structure works."""
    
    def test_class_method(self):
        """Test that class-based tests work."""
        assert True
    
    def test_class_with_fixture(self, mock_config):
        """Test that fixtures work with class-based tests."""
        assert mock_config["debug"] is True


def test_parametrized_test(temp_dir):
    """Test that parametrized tests work."""
    test_files = ["test1.txt", "test2.txt", "test3.txt"]
    
    for filename in test_files:
        test_file = temp_dir / filename
        test_file.write_text(f"Content of {filename}")
        assert test_file.exists()
        assert filename in test_file.read_text()


@pytest.mark.parametrize("input_value,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
])
def test_parametrized_with_decorator(input_value, expected):
    """Test parametrized tests using pytest.mark.parametrize."""
    assert input_value * 2 == expected


def test_exception_handling():
    """Test that exception handling works in tests."""
    with pytest.raises(ValueError):
        raise ValueError("Test exception")
    
    with pytest.raises(ValueError, match="Test exception"):
        raise ValueError("Test exception")


def test_mock_patch_functionality(mock_file_operations):
    """Test that mock patching functionality works."""
    # The fixture should have patched file operations
    assert "open" in mock_file_operations
    assert "exists" in mock_file_operations
    assert "makedirs" in mock_file_operations
    
    # Test that the mocks are working
    mock_file_operations["exists"].return_value = True
    assert mock_file_operations["exists"]("any_path") is True