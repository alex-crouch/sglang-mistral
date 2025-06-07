"""Unit tests for SGLang Mistral client functionality."""

import pytest
from unittest.mock import Mock, patch
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

from sglang_mistral.client import SGLangMistralClient


@pytest.mark.unit
class TestSGLangMistralClientInit:
    """Test cases for SGLangMistralClient initialization."""

    def test_init_with_defaults(self):
        """Test client initialization with default values."""
        client = SGLangMistralClient()

        assert client.host == "localhost"
        assert client.port == 30000
        assert client.model == "OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
        assert client.base_url == "http://localhost:30000"

    def test_init_with_custom_values(self):
        """Test client initialization with custom values."""
        client = SGLangMistralClient(
            host="custom-host",
            port=8080,
            model="custom-model"
        )

        assert client.host == "custom-host"
        assert client.port == 8080
        assert client.model == "custom-model"
        assert client.base_url == "http://custom-host:8080"

    def test_init_with_env_vars(self, mock_env_vars):
        """Test client initialization using environment variables."""
        client = SGLangMistralClient()

        assert client.host == "env-host"
        assert client.port == 8080
        assert client.base_url == "http://env-host:8080"

    def test_init_custom_overrides_env(self, mock_env_vars):
        """Test that custom values override environment variables."""
        client = SGLangMistralClient(host="override-host", port=9999)

        assert client.host == "override-host"
        assert client.port == 9999
        assert client.base_url == "http://override-host:9999"

    def test_init_partial_custom_values(self, mock_env_vars):
        """Test initialization with some custom values and some from env."""
        client = SGLangMistralClient(host="custom-host")

        assert client.host == "custom-host"
        assert client.port == 8080  # From env var
        assert client.base_url == "http://custom-host:8080"


@pytest.mark.unit
class TestSGLangMistralClientMakeRequest:
    """Test cases for make_request method."""

    def test_make_request_success(self, mock_client, mock_requests_post, sample_text_message, mock_successful_response):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        result = mock_client.make_request(sample_text_message)

        assert result == mock_successful_response
        mock_requests_post.assert_called_once_with(
            "http://test-host:9999/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": sample_text_message,
                "max_tokens": 300
            }
        )

    def test_make_request_with_custom_max_tokens(self, mock_client, mock_requests_post, sample_text_message, mock_successful_response):
        """Test API request with custom max_tokens."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        mock_client.make_request(sample_text_message, max_tokens=500)

        mock_requests_post.assert_called_once_with(
            "http://test-host:9999/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": sample_text_message,
                "max_tokens": 500
            }
        )

    def test_make_request_with_kwargs(self, mock_client, mock_requests_post, sample_text_message, mock_successful_response):
        """Test API request with additional keyword arguments."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        mock_client.make_request(
            sample_text_message,
            max_tokens=400,
            temperature=0.7,
            top_p=0.9
        )

        mock_requests_post.assert_called_once_with(
            "http://test-host:9999/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": sample_text_message,
                "max_tokens": 400,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )

    def test_make_request_http_error(self, mock_client, mock_requests_post, sample_text_message):
        """Test API request with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = HTTPError("HTTP 500 Error")
        mock_requests_post.return_value = mock_response

        with pytest.raises(HTTPError):
            mock_client.make_request(sample_text_message)

    def test_make_request_connection_error(self, mock_client, mock_requests_post, sample_text_message):
        """Test API request with connection error."""
        mock_requests_post.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            mock_client.make_request(sample_text_message)

    def test_make_request_timeout_error(self, mock_client, mock_requests_post, sample_text_message):
        """Test API request with timeout error."""
        mock_requests_post.side_effect = Timeout("Request timed out")

        with pytest.raises(Timeout):
            mock_client.make_request(sample_text_message)

    def test_make_request_generic_request_exception(self, mock_client, mock_requests_post, sample_text_message):
        """Test API request with generic request exception."""
        mock_requests_post.side_effect = RequestException("Generic request error")

        with pytest.raises(RequestException):
            mock_client.make_request(sample_text_message)


@pytest.mark.unit
class TestSGLangMistralClientMakeRequestRaw:
    """Test cases for make_request_raw method."""

    def test_make_request_raw_success(self, mock_client, mock_requests_post, sample_text_message):
        """Test successful raw API request."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "Raw response text"
        mock_requests_post.return_value = mock_response

        result = mock_client.make_request_raw(sample_text_message)

        assert result == "Raw response text"
        mock_requests_post.assert_called_once_with(
            "http://test-host:9999/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": sample_text_message,
                "max_tokens": 300
            }
        )

    def test_make_request_raw_with_custom_params(self, mock_client, mock_requests_post, sample_text_message):
        """Test raw API request with custom parameters."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "Custom raw response"
        mock_requests_post.return_value = mock_response

        result = mock_client.make_request_raw(
            sample_text_message,
            max_tokens=1000,
            stream=True
        )

        assert result == "Custom raw response"
        mock_requests_post.assert_called_once_with(
            "http://test-host:9999/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": sample_text_message,
                "max_tokens": 1000,
                "stream": True
            }
        )

    def test_make_request_raw_error(self, mock_client, mock_requests_post, sample_text_message):
        """Test raw API request with error."""
        mock_requests_post.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            mock_client.make_request_raw(sample_text_message)


@pytest.mark.unit
class TestSGLangMistralClientHealthCheck:
    """Test cases for health_check method."""

    def test_health_check_success(self, mock_client, mock_requests_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        result = mock_client.health_check()

        assert result is True
        mock_requests_get.assert_called_once_with(
            "http://test-host:9999/health",
            timeout=5
        )

    def test_health_check_failure_status_code(self, mock_client, mock_requests_get):
        """Test health check with non-200 status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests_get.return_value = mock_response

        result = mock_client.health_check()

        assert result is False

    def test_health_check_connection_error(self, mock_client, mock_requests_get):
        """Test health check with connection error."""
        mock_requests_get.side_effect = ConnectionError("Cannot connect")

        result = mock_client.health_check()

        assert result is False

    def test_health_check_timeout(self, mock_client, mock_requests_get):
        """Test health check with timeout."""
        mock_requests_get.side_effect = Timeout("Request timed out")

        result = mock_client.health_check()

        assert result is False

    def test_health_check_generic_exception(self, mock_client, mock_requests_get):
        """Test health check with generic request exception."""
        mock_requests_get.side_effect = RequestException("Generic error")

        result = mock_client.health_check()

        assert result is False


@pytest.mark.unit
class TestSGLangMistralClientRepr:
    """Test cases for __repr__ method."""

    def test_repr_default_values(self):
        """Test string representation with default values."""
        client = SGLangMistralClient()

        expected = "SGLangMistralClient(host='localhost', port=30000, model='OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym')"
        assert repr(client) == expected

    def test_repr_custom_values(self):
        """Test string representation with custom values."""
        client = SGLangMistralClient(
            host="custom-host",
            port=8080,
            model="custom-model"
        )

        expected = "SGLangMistralClient(host='custom-host', port=8080, model='custom-model')"
        assert repr(client) == expected


@pytest.mark.unit
class TestSGLangMistralClientIntegration:
    """Integration tests combining multiple client methods."""

    def test_client_workflow_success(self, mock_requests_post, mock_requests_get, sample_text_message, mock_successful_response):
        """Test complete client workflow: health check + request."""
        # Setup health check mock
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_requests_get.return_value = mock_health_response

        # Setup API request mock
        mock_api_response = Mock()
        mock_api_response.raise_for_status.return_value = None
        mock_api_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_api_response

        client = SGLangMistralClient(host="integration-test", port=7777)

        # Check health first
        health_ok = client.health_check()
        assert health_ok is True

        # Make request
        result = client.make_request(sample_text_message)
        assert result == mock_successful_response

        # Verify calls
        mock_requests_get.assert_called_once_with(
            "http://integration-test:7777/health",
            timeout=5
        )
        mock_requests_post.assert_called_once()

    def test_client_workflow_health_fail_request_success(self, mock_requests_post, mock_requests_get, sample_text_message, mock_successful_response):
        """Test workflow where health check fails but request succeeds."""
        # Setup health check mock (failure)
        mock_requests_get.side_effect = ConnectionError("Health check failed")

        # Setup API request mock (success)
        mock_api_response = Mock()
        mock_api_response.raise_for_status.return_value = None
        mock_api_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_api_response

        client = SGLangMistralClient()

        # Health check should fail
        health_ok = client.health_check()
        assert health_ok is False

        # But request might still work (different endpoint)
        result = client.make_request(sample_text_message)
        assert result == mock_successful_response

    @patch.dict('os.environ', {'SGLANG_HOST': 'env-test-host', 'SGLANG_PORT': '3333'})
    def test_client_with_env_integration(self, mock_requests_post, sample_text_message, mock_successful_response):
        """Test client using environment variables in integration scenario."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        client = SGLangMistralClient()

        # Client should use environment variables
        assert client.host == "env-test-host"
        assert client.port == 3333
        assert client.base_url == "http://env-test-host:3333"

        # Request should use correct URL
        client.make_request(sample_text_message)

        mock_requests_post.assert_called_once_with(
            "http://env-test-host:3333/v1/chat/completions",
            json={
                "model": client.model,
                "messages": sample_text_message,
                "max_tokens": 300
            }
        )
