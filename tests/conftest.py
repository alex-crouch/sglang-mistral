"""Pytest configuration and fixtures for SGLang Mistral tests."""

import pytest
from unittest.mock import Mock, patch
import os

from sglang_mistral.client import SGLangMistralClient
from sglang_mistral.message import create_message


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("SGLANG_HOST", "env-host")
    monkeypatch.setenv("SGLANG_PORT", "8080")
    monkeypatch.setenv("SGLANG_MODEL", "env-model")


@pytest.fixture
def mock_client():
    """Create a real SGLangMistralClient instance for testing."""
    return SGLangMistralClient(
        host="test-host",
        port=9999,
        model="test-model"
    )


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for HTTP requests."""
    with patch("requests.post") as mock_post:
        yield mock_post


@pytest.fixture
def sample_text_message():
    """Provide a sample text message for testing."""
    return create_message("What can you see in this image?")


@pytest.fixture
def sample_image_message():
    """Provide a sample message with image for testing."""
    return create_message("Describe this image", ["http://example.com/image.jpg"])


@pytest.fixture
def sample_multi_image_message():
    """Provide a sample message with multiple images for testing."""
    return create_message("Compare these images", [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg"
    ])


@pytest.fixture
def mock_successful_response():
    """Mock successful API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the API."
                }
            }
        ]
    }


@pytest.fixture
def mock_health_response():
    """Mock health check response."""
    return {"status": "ok"}


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for health check requests."""
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_response():
    """Generic mock response object."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"test": "data"}
    response.status_code = 200
    return response


@pytest.fixture
def mock_error_response():
    """Mock error response object."""
    response = Mock()
    response.status_code = 500
    response.text = "Internal Server Error"
    return response


@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary .env file for testing."""
    env_file = tmp_path / ".env"
    env_content = """SGLANG_HOST=test-host
SGLANG_PORT=9999
SGLANG_MODEL=test-model
"""
    env_file.write_text(env_content)
    return str(env_file)


@pytest.fixture
def default_image_url():
    """Provide a default image URL for testing."""
    return "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    env_vars = ["SGLANG_HOST", "SGLANG_PORT", "SGLANG_MODEL"]
    original_values = {}

    # Store original values
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]
