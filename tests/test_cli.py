"""Unit tests for CLI functionality."""

import pytest
from unittest.mock import Mock, patch
from io import StringIO

from sglang_mistral.cli import (
    create_parser,
    determine_text_and_images,
    main
)


@pytest.mark.unit
class TestCreateParser:
    """Test cases for create_parser function."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.description == "Interact with SGLang Mistral model"

    def test_parser_help(self):
        """Test parser help output."""
        parser = create_parser()
        help_text = parser.format_help()

        # Check that key arguments are documented
        assert "--host" in help_text
        assert "--port" in help_text
        assert "--model" in help_text
        assert "--text" in help_text
        assert "--image-url" in help_text
        assert "--max-tokens" in help_text

    def test_parser_examples_in_help(self):
        """Test that examples are included in help."""
        parser = create_parser()
        help_text = parser.format_help()

        assert "Examples:" in help_text
        assert "--text" in help_text
        assert "--image-url" in help_text

    def test_parse_text_only(self):
        """Test parsing text-only arguments."""
        parser = create_parser()
        args = parser.parse_args(["--text", "Hello world"])

        assert args.text == "Hello world"
        assert args.image_url is None
        assert args.max_tokens == 300

    def test_parse_image_only(self):
        """Test parsing image-only arguments."""
        parser = create_parser()
        args = parser.parse_args(["--image-url", "https://example.com/img.jpg"])

        assert args.text is None
        assert args.image_url == [["https://example.com/img.jpg"]]

    def test_parse_multiple_images(self):
        """Test parsing multiple image URLs."""
        parser = create_parser()
        args = parser.parse_args([
            "--image-url", "https://example.com/img1.jpg",
            "--image-url", "https://example.com/img2.jpg",
            "--image-url", "https://example.com/img3.jpg"
        ])

        assert args.image_url == [
            ["https://example.com/img1.jpg"],
            ["https://example.com/img2.jpg"],
            ["https://example.com/img3.jpg"]
        ]

    def test_parse_multiple_images_new_syntax(self):
        """Test parsing multiple image URLs with new syntax (single flag)."""
        parser = create_parser()
        args = parser.parse_args([
            "--image-url", "https://example.com/img1.jpg", "https://example.com/img2.jpg", "https://example.com/img3.jpg"
        ])

        # With nargs='*' and action='append', this creates nested lists
        assert args.image_url == [
            ["https://example.com/img1.jpg", "https://example.com/img2.jpg", "https://example.com/img3.jpg"]
        ]

    def test_parse_text_and_images(self):
        """Test parsing both text and images."""
        parser = create_parser()
        args = parser.parse_args([
            "--text", "Compare these",
            "--image-url", "https://example.com/img1.jpg",
            "--image-url", "https://example.com/img2.jpg"
        ])

        assert args.text == "Compare these"
        assert args.image_url == [
            ["https://example.com/img1.jpg"],
            ["https://example.com/img2.jpg"]
        ]

    def test_parse_server_config(self):
        """Test parsing server configuration."""
        parser = create_parser()
        args = parser.parse_args([
            "--host", "custom-host",
            "--port", "8080",
            "--model", "custom-model"
        ])

        assert args.host == "custom-host"
        assert args.port == 8080
        assert args.model == "custom-model"

    def test_parse_max_tokens(self):
        """Test parsing max tokens argument."""
        parser = create_parser()
        args = parser.parse_args(["--max-tokens", "500"])

        assert args.max_tokens == 500

    def test_parse_output_options(self):
        """Test parsing output format options."""
        parser = create_parser()

        # Test raw option
        args_raw = parser.parse_args(["--raw"])
        assert args_raw.raw is True
        assert args_raw.json is False

        # Test json option
        args_json = parser.parse_args(["--json"])
        assert args_json.raw is False
        assert args_json.json is True

    def test_parse_host_alias(self):
        """Test that --addr is an alias for --host."""
        parser = create_parser()
        args = parser.parse_args(["--addr", "alias-host"])

        assert args.host == "alias-host"

    def test_parse_no_args(self):
        """Test parsing with no arguments (all defaults)."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.text is None
        assert args.image_url is None
        assert args.host is None
        assert args.port is None
        assert args.model is None
        assert args.max_tokens == 300
        assert args.raw is False
        assert args.json is False


@pytest.mark.unit
class TestDetermineTextAndImages:
    """Test cases for determine_text_and_images function."""

    def test_custom_text_no_images(self):
        """Test with custom text and no images."""
        text, images = determine_text_and_images("Custom text", None)

        assert text == "Custom text"
        assert images is None

    def test_custom_text_with_images(self):
        """Test with custom text and images."""
        image_urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
        text, images = determine_text_and_images("Custom text", image_urls)

        assert text == "Custom text"
        assert images == image_urls

    def test_no_text_single_image(self):
        """Test with no text and single image."""
        image_urls = ["https://example.com/img.jpg"]
        text, images = determine_text_and_images(None, image_urls)

        assert text == "What's in this image?"
        assert images == image_urls

    def test_no_text_multiple_images(self):
        """Test with no text and multiple images."""
        image_urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
        text, images = determine_text_and_images(None, image_urls)

        assert text == "I have 2 images. Please describe each image."
        assert images == image_urls

    def test_no_text_many_images(self):
        """Test with no text and many images."""
        image_urls = [f"https://example.com/img{i}.jpg" for i in range(5)]
        text, images = determine_text_and_images(None, image_urls)

        assert text == "I have 5 images. Please describe each image."
        assert images == image_urls

    def test_no_text_no_images(self):
        """Test with no text and no images (default case)."""
        text, images = determine_text_and_images(None, None)

        assert text == "What's in this image?"
        assert images == ["https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"]

    def test_empty_text_with_images(self):
        """Test with empty text and images - empty string is preserved as provided."""
        image_urls = ["https://example.com/img.jpg"]
        text, images = determine_text_and_images("", image_urls)

        # Empty string is explicitly provided, so it should be preserved
        assert text == ""
        assert images == image_urls

    def test_empty_image_list(self):
        """Test with empty image list."""
        text, images = determine_text_and_images("Custom text", [])

        assert text == "Custom text"
        assert images == []


@pytest.mark.unit
class TestMainFunction:
    """Test cases for main function."""

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_text_only_success(self, mock_client_class, mock_create_parser):
        """Test main function with text-only request."""
        # Setup mocks
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = "Hello world"
        mock_args.image_url = None
        mock_args.host = None
        mock_args.port = None
        mock_args.model = None
        mock_args.max_tokens = 300
        mock_args.raw = False
        mock_args.json = False

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "test"}
        mock_client_class.return_value = mock_client

        # Capture stdout
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = main()

        assert result == 0
        mock_client.make_request.assert_called_once()
        assert "{'response': 'test'}" in mock_stdout.getvalue()

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_with_images_success(self, mock_client_class, mock_create_parser):
        """Test main function with image request."""
        # Setup mocks
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = None
        mock_args.image_url = ["https://example.com/img.jpg"]
        mock_args.host = "test-host"
        mock_args.port = 8080
        mock_args.model = "test-model"
        mock_args.max_tokens = 500
        mock_args.raw = False
        mock_args.json = False

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "image analysis"}
        mock_client_class.return_value = mock_client

        with patch('sys.stdout', new_callable=StringIO):
            result = main()

        assert result == 0
        mock_client_class.assert_called_once_with(
            host="test-host",
            port=8080,
            model="test-model"
        )

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_raw_output(self, mock_client_class, mock_create_parser):
        """Test main function with raw output."""
        # Setup mocks
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = "Test"
        mock_args.image_url = None
        mock_args.host = None
        mock_args.port = None
        mock_args.model = None
        mock_args.max_tokens = 300
        mock_args.raw = True
        mock_args.json = False

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_client = Mock()
        mock_client.make_request_raw.return_value = "raw response text"
        mock_client_class.return_value = mock_client

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = main()

        assert result == 0
        mock_client.make_request_raw.assert_called_once()
        assert "raw response text" in mock_stdout.getvalue()

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_json_output(self, mock_client_class, mock_create_parser):
        """Test main function with JSON output."""
        # Setup mocks
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = "Test"
        mock_args.image_url = None
        mock_args.host = None
        mock_args.port = None
        mock_args.model = None
        mock_args.max_tokens = 300
        mock_args.raw = False
        mock_args.json = True

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "formatted"}
        mock_client_class.return_value = mock_client

        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = main()

        assert result == 0
        output = mock_stdout.getvalue()
        # Should be formatted JSON
        assert '"response": "formatted"' in output
        assert "{" in output and "}" in output

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_client_error(self, mock_client_class, mock_create_parser):
        """Test main function with client error."""
        # Setup mocks
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = "Test"
        mock_args.image_url = None
        mock_args.host = None
        mock_args.port = None
        mock_args.model = None
        mock_args.max_tokens = 300
        mock_args.raw = False
        mock_args.json = False

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_client = Mock()
        mock_client.make_request.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()

        assert result == 1
        assert "Error: Connection error" in mock_stderr.getvalue()

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_keyboard_interrupt(self, mock_client_class, mock_create_parser):
        """Test main function with keyboard interrupt."""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = "Test"
        mock_args.image_url = None
        mock_args.host = None
        mock_args.port = None
        mock_args.model = None
        mock_args.max_tokens = 300
        mock_args.raw = False
        mock_args.json = False

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        # Simulate KeyboardInterrupt during client creation
        mock_client_class.side_effect = KeyboardInterrupt()

        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            result = main()

        assert result == 130
        assert "Interrupted by user" in mock_stderr.getvalue()

    @patch('sglang_mistral.cli.create_parser')
    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_main_default_case(self, mock_client_class, mock_create_parser):
        """Test main function with default case (no args)."""
        # Setup mocks for default case
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.text = None
        mock_args.image_url = None
        mock_args.host = None
        mock_args.port = None
        mock_args.model = None
        mock_args.max_tokens = 300
        mock_args.raw = False
        mock_args.json = False

        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser

        mock_client = Mock()
        mock_client.make_request.return_value = {"response": "default"}
        mock_client_class.return_value = mock_client

        with patch('sys.stdout', new_callable=StringIO):
            result = main()

        assert result == 0
        # Should have called make_request with default image
        mock_client.make_request.assert_called_once()
        call_args = mock_client.make_request.call_args
        # Check that messages were created (first argument)
        assert call_args[1]['messages'] is not None


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_cli_workflow_text_only(self, mock_client_class):
        """Test complete CLI workflow for text-only request."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"test": "response"}
        mock_client_class.return_value = mock_client

        # Simulate command line args
        test_args = ["--text", "Hello world", "--host", "test-host"]

        with patch('sys.argv', ['cli_test'] + test_args):
            with patch('sys.stdout', new_callable=StringIO):
                result = main()

        assert result == 0
        mock_client_class.assert_called_once_with(
            host="test-host",
            port=None,
            model=None
        )

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_cli_workflow_multiple_images(self, mock_client_class):
        """Test complete CLI workflow for multiple images."""
        mock_client = Mock()
        mock_client.make_request.return_value = {"analysis": "multiple images"}
        mock_client_class.return_value = mock_client

        test_args = [
            "--text", "Compare these",
            "--image-url", "https://example.com/img1.jpg",
            "--image-url", "https://example.com/img2.jpg",
            "--max-tokens", "500"
        ]

        with patch('sys.argv', ['cli_test'] + test_args):
            with patch('sys.stdout', new_callable=StringIO):
                result = main()

        assert result == 0
        mock_client.make_request.assert_called_once()
        call_kwargs = mock_client.make_request.call_args[1]
        assert call_kwargs['max_tokens'] == 500

    @patch('sglang_mistral.cli.SGLangMistralClient')
    def test_image_url_flattening_integration(self, mock_client_class):
        """Test that image URL flattening works for both old and new syntax."""
        from unittest.mock import Mock, patch
        from io import StringIO

        mock_client = Mock()
        mock_client.make_request.return_value = {"test": "response"}
        mock_client_class.return_value = mock_client

        # Test new syntax: single --image-url with multiple URLs
        test_args_new = [
            "--text", "Analyze these images",
            "--image-url", "https://example.com/img1.jpg", "https://example.com/img2.jpg"
        ]

        with patch('sys.argv', ['cli_test'] + test_args_new):
            with patch('sys.stdout', new_callable=StringIO):
                result = main()

        assert result == 0
        mock_client.make_request.assert_called()

        # Get the messages that were passed to make_request
        call_args = mock_client.make_request.call_args
        messages = call_args[1]['messages']

        # Should have 3 content items: 2 images + text (images come first)
        assert len(messages[0]['content']) == 3
        assert messages[0]['content'][0]['type'] == 'image_url'
        assert messages[0]['content'][1]['type'] == 'image_url'
        assert messages[0]['content'][2]['type'] == 'text'
        assert messages[0]['content'][0]['image_url']['url'] == "https://example.com/img1.jpg"
        assert messages[0]['content'][1]['image_url']['url'] == "https://example.com/img2.jpg"

    def test_determine_text_and_images_integration(self):
            """Test integration between determine_text_and_images and message creation."""
            # Test case 1: Custom text with images
            text1, images1 = determine_text_and_images(
                "Custom analysis",
                ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
            )
            assert text1 == "Custom analysis"
            assert images1 is not None
            assert len(images1) == 2

            # Test case 2: No text, multiple images
            text2, images2 = determine_text_and_images(
                None,
                ["https://example.com/img1.jpg", "https://example.com/img2.jpg", "https://example.com/img3.jpg"]
            )
            assert "3 images" in text2
            assert images2 is not None
            assert len(images2) == 3

            # Test case 3: Default case
            text3, images3 = determine_text_and_images(None, None)
            assert text3 == "What's in this image?"
            assert images3 is not None
            assert len(images3) == 1
            assert "github.com" in images3[0]
