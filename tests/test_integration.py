"""Integration tests for SGLang Mistral client."""

import pytest
from unittest.mock import Mock, patch
from requests.exceptions import ConnectionError

from sglang_mistral.client import SGLangMistralClient
from sglang_mistral.message import create_message, determine_default_text
from sglang_mistral.cli import determine_text_and_images, main


@pytest.mark.integration
class TestClientMessageIntegration:
    """Integration tests combining client and message functionality."""

    def test_client_with_text_message(self, mock_requests_post, mock_successful_response):
        """Test client with text-only message creation."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        client = SGLangMistralClient(host="test-host", port=8080)
        messages = create_message("Hello, world!")

        result = client.make_request(messages)

        assert result == mock_successful_response

        # Verify the request was made with correct structure
        call_args = mock_requests_post.call_args
        request_data = call_args[1]['json']
        assert request_data['messages'] == [{"role": "user", "content": "Hello, world!"}]

    def test_client_with_single_image_message(self, mock_requests_post, mock_successful_response):
        """Test client with single image message creation."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        client = SGLangMistralClient()
        messages = create_message("What's in this image?", ["https://example.com/test.jpg"])

        result = client.make_request(messages, max_tokens=500)

        assert result == mock_successful_response

        # Verify the request structure
        call_args = mock_requests_post.call_args
        request_data = call_args[1]['json']
        assert request_data['max_tokens'] == 500
        assert len(request_data['messages'][0]['content']) == 2
        assert request_data['messages'][0]['content'][0]['type'] == 'image_url'
        assert request_data['messages'][0]['content'][1]['type'] == 'text'

    def test_client_with_multiple_images_message(self, mock_requests_post, mock_successful_response):
        """Test client with multiple images message creation."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_successful_response
        mock_requests_post.return_value = mock_response

        client = SGLangMistralClient()
        image_urls = [
            "https://example.com/img1.jpg",
            "https://example.com/img2.jpg",
            "https://example.com/img3.jpg"
        ]
        messages = create_message("Analyze these images", image_urls)

        client.make_request(messages)

        # Verify the request structure for multiple images
        call_args = mock_requests_post.call_args
        request_data = call_args[1]['json']
        content = request_data['messages'][0]['content']

        assert len(content) == 4  # 3 images + 1 text
        for i in range(3):
            assert content[i]['type'] == 'image_url'
            assert content[i]['image_url']['url'] == image_urls[i]
        assert content[3]['type'] == 'text'
        assert content[3]['text'] == "Analyze these images"


@pytest.mark.integration
class TestCLIMessageIntegration:
    """Integration tests combining CLI and message functionality."""

    def test_cli_text_determination_with_message_creation(self):
        """Test CLI text determination integrated with message creation."""
        # Case 1: No text, single image
        text, images = determine_text_and_images(None, ["https://example.com/img.jpg"])
        messages = create_message(text, images)

        assert text == "What's in this image?"
        assert len(messages[0]['content']) == 2
        assert messages[0]['content'][1]['text'] == "What's in this image?"

        # Case 2: No text, multiple images
        text, images = determine_text_and_images(None, ["img1.jpg", "img2.jpg", "img3.jpg"])
        messages = create_message(text, images)

        assert "3 images" in text
        assert len(messages[0]['content']) == 4
        assert "3 images" in messages[0]['content'][3]['text']

        # Case 3: Custom text with images
        text, images = determine_text_and_images("Custom prompt", ["img1.jpg", "img2.jpg"])
        messages = create_message(text, images)

        assert text == "Custom prompt"
        assert messages[0]['content'][2]['text'] == "Custom prompt"

    def test_default_text_determination_integration(self):
        """Test default text determination with various image counts."""
        test_cases = [
            (0, "What's in this image?"),
            (1, "What's in this image?"),
            (2, "I have 2 images. Please describe each image."),
            (5, "I have 5 images. Please describe each image."),
        ]

        for count, expected_text in test_cases:
            determined_text = determine_default_text(count)
            assert determined_text == expected_text

            # Test with actual message creation
            if count > 0:
                image_urls = [f"https://example.com/img{i}.jpg" for i in range(count)]
                messages = create_message(determined_text, image_urls)

                if count == 1:
                    # Single image: simple structure might be used, but we use complex for consistency
                    assert len(messages[0]['content']) == 2
                else:
                    # Multiple images
                    assert len(messages[0]['content']) == count + 1


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end workflow tests without external dependencies."""

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_complete_text_workflow(self, mock_client_class):
        """Test complete workflow for text-only request."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "Hello back!"}
        mock_client_class.return_value = mock_client

        # Simulate CLI call
        with patch('sys.argv', ['test', '--text', 'Hello world']):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0
        mock_client.make_request.assert_called_once()

        # Verify the message structure passed to client
        call_args = mock_client.make_request.call_args
        messages = call_args[1]['messages']
        assert messages == [{"role": "user", "content": "Hello world"}]

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_complete_image_workflow(self, mock_client_class):
        """Test complete workflow for image request."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "I see an image"}
        mock_client_class.return_value = mock_client

        test_url = "https://example.com/test-image.jpg"

        with patch('sys.argv', ['test', '--image-url', test_url]):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0

        # Verify the message structure
        call_args = mock_client.make_request.call_args
        messages = call_args[1]['messages']
        content = messages[0]['content']

        assert len(content) == 2
        assert content[0]['type'] == 'image_url'
        assert content[0]['image_url']['url'] == test_url
        assert content[1]['type'] == 'text'
        assert content[1]['text'] == "What's in this image?"

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_complete_multiple_images_workflow(self, mock_client_class):
        """Test complete workflow for multiple images."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "Multiple images analyzed"}
        mock_client_class.return_value = mock_client

        test_urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]

        with patch('sys.argv', [
            'test',
            '--text', 'Compare these images',
            '--image-url', test_urls[0],
            '--image-url', test_urls[1]
        ]):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0

        # Verify the message structure
        call_args = mock_client.make_request.call_args
        messages = call_args[1]['messages']
        content = messages[0]['content']

        assert len(content) == 3  # 2 images + 1 text
        assert content[0]['image_url']['url'] == test_urls[0]
        assert content[1]['image_url']['url'] == test_urls[1]
        assert content[2]['text'] == 'Compare these images'

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_complete_default_workflow(self, mock_client_class, default_image_url):
        """Test complete workflow with no arguments (default case)."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "Default response"}
        mock_client_class.return_value = mock_client

        with patch('sys.argv', ['test']):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0

        # Verify default behavior
        call_args = mock_client.make_request.call_args
        messages = call_args[1]['messages']
        content = messages[0]['content']

        assert len(content) == 2
        assert content[0]['type'] == 'image_url'
        assert content[0]['image_url']['url'] == default_image_url
        assert content[1]['text'] == "What's in this image?"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_client_error_propagation_through_cli(self, mock_client_class):
            """Test that client errors are properly handled by CLI."""
            mock_client = Mock()
            mock_client.make_request.side_effect = ConnectionError("Cannot connect to server")
            mock_client_class.return_value = mock_client

            with patch('sys.argv', ['test', '--text', 'Hello']):
                with patch('sys.stderr', new=Mock()) as mock_stderr:
                    result = main()

            assert result == 1
            # Error should be caught and handled by CLI
            mock_stderr.write.assert_called()

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_invalid_message_handling(self, mock_client_class):
        """Test handling of edge cases in message creation through CLI."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "OK"}
        mock_client_class.return_value = mock_client

        # Test with empty text
        with patch('sys.argv', ['test', '--text', '']):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0

        # Verify empty text was handled
        call_args = mock_client.make_request.call_args
        messages = call_args[1]['messages']
        assert messages[0]['content'] == ""

    def test_message_creation_edge_cases_integration(self):
        """Test edge cases in message creation that might affect integration."""
        # Very long text
        long_text = "A" * 5000
        messages = create_message(long_text)
        assert messages[0]['content'] == long_text

        # Special characters
        special_text = "🎉 Hello! Special chars: @#$%^&*()"
        messages = create_message(special_text)
        assert messages[0]['content'] == special_text

        # Empty and invalid URLs
        messages = create_message("Test", ["", "https://valid.com/img.jpg"])
        content = messages[0]['content']
        assert len(content) == 3  # empty URL + valid URL + text
        assert content[0]['image_url']['url'] == ""
        assert content[1]['image_url']['url'] == "https://valid.com/img.jpg"


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration across components."""

    @patch.dict('os.environ', {'SGLANG_HOST': 'env-host', 'SGLANG_PORT': '9999'})
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_environment_variable_integration(self, mock_client_class):
        """Test that environment variables work through the complete stack."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "OK"}
        mock_client_class.return_value = mock_client

        with patch('sys.argv', ['test', '--text', 'Hello']):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0

        # Verify client was created with environment variables
        mock_client_class.assert_called_once_with(
            host=None,  # CLI passes None, client reads from env
            port=None,  # CLI passes None, client reads from env
            model=None
        )

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_cli_argument_override_integration(self, mock_client_class):
        """Test that CLI arguments properly override defaults."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "OK"}
        mock_client_class.return_value = mock_client

        with patch('sys.argv', [
            'test',
            '--host', 'cli-host',
            '--port', '7777',
            '--model', 'cli-model',
            '--max-tokens', '1000',
            '--text', 'Hello'
        ]):
            with patch('sys.stdout', new=Mock()):
                result = main()

        assert result == 0

        # Verify client configuration
        mock_client_class.assert_called_once_with(
            host='cli-host',
            port=7777,
            model='cli-model'
        )

        # Verify request configuration
        call_args = mock_client.make_request.call_args
        assert call_args[1]['max_tokens'] == 1000


@pytest.mark.integration
class TestOutputFormatIntegration:
    """Integration tests for different output formats."""

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_raw_output_integration(self, mock_client_class):
            """Test raw output format through complete workflow."""
            mock_client = Mock()
            mock_client.make_request_raw.return_value = "Raw response from server"
            mock_client_class.return_value = mock_client

            with patch('sys.argv', ['test', '--text', 'Hello', '--raw']):
                with patch('sys.stdout', new=Mock()):
                    result = main()

            assert result == 0
            mock_client.make_request_raw.assert_called_once()

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_json_output_integration(self, mock_client_class):
            """Test JSON output format through complete workflow."""
            mock_client = Mock()
            mock_client.make_request.return_value = {"response": "test", "tokens": 42}
            mock_client_class.return_value = mock_client

            with patch('sys.argv', ['test', '--text', 'Hello', '--json']):
                with patch('sys.stdout', new=Mock()):
                    result = main()

            assert result == 0
            mock_client.make_request.assert_called_once()
