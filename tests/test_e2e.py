"""End-to-end tests against actual SGLang server at localhost:30000."""

import os
import pytest
import requests
import json
from unittest.mock import patch

from sglang_mistral.client import SGLangMistralClient
from sglang_mistral.message import create_message
from sglang_mistral.cli import main


def is_server_available(timeout=5):
    """Check if SGLang server is available."""
    host = os.getenv("SGLANG_HOST", "localhost")
    port = int(os.getenv("SGLANG_PORT", "30000"))
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


pytestmark = pytest.mark.skipif(
    not is_server_available(),
    reason="SGLang server not available"
)


@pytest.mark.e2e
class TestServerTextRequests:
    """Test text-only requests against real SGLang server."""

    def setup_method(self):
        """Set up test with real client."""
        self.client = SGLangMistralClient()  # Let it read from env

    def test_simple_text_request(self):
        """Test basic text request to real server."""
        messages = create_message("Hello, can you respond with 'Test successful'?")

        response = self.client.make_request(messages, max_tokens=50)

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]

        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

    def test_text_request_with_custom_max_tokens(self):
        """Test text request with custom max_tokens."""
        messages = create_message("Write a very short poem about AI.")

        response = self.client.make_request(messages, max_tokens=100)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

    def test_text_request_raw_response(self):
        """Test raw text response from server."""
        messages = create_message("Respond with exactly: 'Raw response test'")

        raw_response = self.client.make_request_raw(messages, max_tokens=30)

        assert isinstance(raw_response, str)
        assert len(raw_response) > 0
        # Should be valid JSON
        parsed = json.loads(raw_response)
        assert "choices" in parsed


@pytest.mark.e2e
class TestServerImageRequests:
    """Test image analysis requests against real SGLang server."""

    def setup_method(self):
        """Set up test with real client."""
        self.client = SGLangMistralClient()  # Let it read from env

    def test_single_image_analysis(self):
        """Test single image analysis request."""
        # Using a simple, publicly available test image
        image_url = "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"
        messages = create_message("What do you see in this image?", [image_url])

        response = self.client.make_request(messages, max_tokens=150)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0
        # Should mention something about an image or visual content
        content_lower = content.lower()
        assert any(word in content_lower for word in ["image", "see", "picture", "visual", "show"])

    def test_multiple_images_analysis(self):
        """Test multiple image analysis request."""
        image_urls = [
            "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true",
            "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"
        ]
        messages = create_message("Compare these two images.", image_urls)

        response = self.client.make_request(messages, max_tokens=200)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

    def test_image_with_specific_question(self):
        """Test image analysis with specific question."""
        image_url = "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"
        messages = create_message("Describe the colors you can see in this image.", [image_url])

        response = self.client.make_request(messages, max_tokens=100)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0


@pytest.mark.e2e
class TestServerCLIIntegration:
    """Test CLI integration with real SGLang server."""

    def test_cli_text_only_real_server(self, capsys):
        """Test CLI with text-only request to real server."""
        test_args = ['test-cli', '--text', 'Hello from CLI test', '--max-tokens', '30']

        with patch('sys.argv', test_args):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_cli_with_image_real_server(self, capsys):
        """Test CLI with image request to real server."""
        image_url = "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"
        test_args = [
            'test-cli',
            '--text', 'What is in this image?',
            '--image-url', image_url,
            '--max-tokens', '50'
        ]

        with patch('sys.argv', test_args):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_cli_with_multiple_images_new_syntax_real_server(self, capsys):
        """Test CLI with multiple images using new syntax (single --image-url flag)."""
        image_url1 = "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"
        image_url2 = "https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"
        test_args = [
            'test-cli',
            '--text', 'Compare these images',
            '--image-url', image_url1, image_url2,
            '--max-tokens', '75'
        ]

        with patch('sys.argv', test_args):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_cli_json_output_real_server(self, capsys):
        """Test CLI with JSON output format."""
        test_args = ['test-cli', '--text', 'Brief response test', '--json', '--max-tokens', '20']

        with patch('sys.argv', test_args):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = captured.out.strip()

        # Should be valid JSON
        parsed_output = json.loads(output)
        assert "choices" in parsed_output

    def test_cli_raw_output_real_server(self, capsys):
        """Test CLI with raw output format."""
        test_args = ['test-cli', '--text', 'Raw output test', '--raw', '--max-tokens', '20']

        with patch('sys.argv', test_args):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        output = captured.out.strip()

        # Raw output should be valid JSON string
        parsed_output = json.loads(output)
        assert "choices" in parsed_output


@pytest.mark.e2e
class TestServerHealthAndErrors:
    """Test server health check and error handling."""

    def setup_method(self):
        """Set up test with real client."""
        self.client = SGLangMistralClient()  # Let it read from env

    def test_health_check_real_server(self):
        """Test health check against real server."""
        is_healthy = self.client.health_check()
        assert is_healthy is True

    def test_server_response_structure(self):
        """Test that server responses have expected structure."""
        messages = create_message("Test response structure")

        response = self.client.make_request(messages, max_tokens=30)

        # Verify response structure matches OpenAI-like format
        assert isinstance(response, dict)
        assert "choices" in response
        assert isinstance(response["choices"], list)
        assert len(response["choices"]) > 0

        choice = response["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]

    def test_invalid_server_port(self):
        """Test behavior with invalid server port."""
        invalid_client = SGLangMistralClient(host="localhost", port=99999)

        # Health check should fail
        is_healthy = invalid_client.health_check()
        assert is_healthy is False

        # Request should raise exception
        messages = create_message("This should fail")
        with pytest.raises(requests.exceptions.RequestException):
            invalid_client.make_request(messages)


@pytest.mark.e2e
class TestServerPerformance:
    """Test basic performance characteristics of real server."""

    def setup_method(self):
        """Set up test with real client."""
        self.client = SGLangMistralClient()  # Let it read from env

    def test_response_time_reasonable(self):
        """Test that server responds within reasonable time."""
        import time

        messages = create_message("Quick response test")

        start_time = time.time()
        response = self.client.make_request(messages, max_tokens=20)
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond within 30 seconds for a simple request
        assert response_time < 30.0
        assert "choices" in response

    def test_concurrent_requests(self):
        """Test server can handle multiple concurrent requests."""
        import threading
        import queue

        def make_request(result_queue, request_id):
            try:
                messages = create_message(f"Concurrent test {request_id}")
                response = self.client.make_request(messages, max_tokens=20)
                result_queue.put(("success", request_id, response))
            except Exception as e:
                result_queue.put(("error", request_id, str(e)))

        result_queue = queue.Queue()
        threads = []

        # Start 3 concurrent requests
        for i in range(3):
            thread = threading.Thread(target=make_request, args=(result_queue, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout per thread

        # Check results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == 3

        # All requests should succeed
        for status, request_id, response in results:
            assert status == "success"
            assert "choices" in response


@pytest.mark.e2e
class TestServerEdgeCases:
    """Test edge cases and boundary conditions with real server."""

    def setup_method(self):
        """Set up test with real client."""
        self.client = SGLangMistralClient()  # Let it read from env

    def test_empty_text_request(self):
        """Test server handling of empty text."""
        messages = create_message("")

        response = self.client.make_request(messages, max_tokens=50)

        assert "choices" in response
        # Server should handle empty text gracefully
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)

    def test_very_long_text_request(self):
        """Test server handling of long text input."""
        long_text = "This is a test with repeated text. " * 50  # ~1750 characters
        messages = create_message(f"{long_text} Please respond briefly.")

        response = self.client.make_request(messages, max_tokens=50)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0

    def test_minimal_max_tokens(self):
        """Test server with minimal max_tokens setting."""
        messages = create_message("Hi")

        response = self.client.make_request(messages, max_tokens=1)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        # Even with max_tokens=1, should get some response

    def test_special_characters_in_text(self):
        """Test server handling of special characters."""
        special_text = "Test with émojis 🚀 and spëcial chars: <>&\"'"
        messages = create_message(special_text)

        response = self.client.make_request(messages, max_tokens=30)

        assert "choices" in response
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content.strip()) > 0
