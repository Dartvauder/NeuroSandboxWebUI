"""Shared pytest fixtures and configuration for the testing suite."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing."""
    temp_file_path = temp_dir / "test_file.tmp"
    temp_file_path.touch()
    try:
        yield temp_file_path
    finally:
        if temp_file_path.exists():
            temp_file_path.unlink()


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "app_name": "test_app",
        "debug": True,
        "version": "0.1.0",
        "paths": {
            "input": "inputs/",
            "output": "outputs/",
            "temp": "temp/",
        },
        "models": {
            "default_model": "test_model",
            "max_workers": 4,
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
        }
    }


@pytest.fixture
def mock_logger() -> Mock:
    """Provide a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def mock_file_operations():
    """Mock file I/O operations for testing."""
    mock_open = MagicMock()
    mock_exists = Mock()
    mock_makedirs = Mock()
    
    mock_open.return_value.__enter__.return_value.read.return_value = "test content"
    mock_exists.return_value = True
    
    return {
        "open": mock_open,
        "exists": mock_exists,
        "makedirs": mock_makedirs,
    }


@pytest.fixture
def sample_image_data() -> bytes:
    """Provide sample image data for testing."""
    # Simple 1x1 PNG image data
    return (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
        b'\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pytest.fixture
def sample_audio_data() -> bytes:
    """Provide sample audio data for testing."""
    # Simple WAV header for testing
    return (
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    )


@pytest.fixture
def sample_video_metadata() -> Dict[str, Any]:
    """Provide sample video metadata for testing."""
    return {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "duration": 10.5,
        "format": "mp4",
        "codec": "h264",
        "bitrate": 5000000,
    }


@pytest.fixture
def mock_model_response() -> Dict[str, Any]:
    """Provide a mock AI model response for testing."""
    return {
        "status": "success",
        "result": {
            "text": "Generated text response",
            "confidence": 0.95,
            "tokens_used": 150,
        },
        "metadata": {
            "model": "test_model",
            "timestamp": "2023-01-01T00:00:00Z",
            "processing_time": 1.23,
        }
    }


@pytest.fixture
def mock_api_client():
    """Provide a mock API client for testing external service calls."""
    client = Mock()
    client.get.return_value = Mock(status_code=200, json=lambda: {"status": "ok"})
    client.post.return_value = Mock(status_code=201, json=lambda: {"id": "test_id"})
    client.put.return_value = Mock(status_code=200, json=lambda: {"updated": True})
    client.delete.return_value = Mock(status_code=204)
    return client


@pytest.fixture
def environment_variables():
    """Temporarily set environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        "TEST_MODE": "true",
        "DEBUG": "1",
        "API_KEY": "test_api_key",
        "MODEL_PATH": "/test/models",
    }
    
    os.environ.update(test_env)
    
    try:
        yield test_env
    finally:
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup any test files after each test."""
    yield
    # Cleanup any test files that might have been created
    test_patterns = ["test_*.tmp", "*.test", "temp_*"]
    for pattern in test_patterns:
        for file_path in Path(".").glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            except (OSError, PermissionError):
                pass


@pytest.fixture
def mock_gradio_interface():
    """Mock Gradio interface for testing UI components."""
    interface = Mock()
    interface.launch = Mock()
    interface.close = Mock()
    interface.predict = Mock(return_value=["test_output"])
    return interface


@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model for testing ML components."""
    model = Mock()
    model.eval = Mock(return_value=model)
    model.train = Mock(return_value=model)
    model.forward = Mock(return_value=Mock(shape=(1, 3, 224, 224)))
    model.load_state_dict = Mock()
    model.state_dict = Mock(return_value={})
    return model


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent