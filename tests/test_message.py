"""Unit tests for message creation functionality."""

import pytest
from typing import List, Optional

from sglang_mistral.message import create_message, determine_default_text


@pytest.mark.unit
class TestCreateMessage:
    """Test cases for create_message function."""

    def test_text_only_message(self):
        """Test creating a text-only message."""
        text = "Hello, world!"
        result = create_message(text)

        expected = [{"role": "user", "content": "Hello, world!"}]
        assert result == expected

    def test_text_only_with_none_images(self):
        """Test creating a text-only message with explicit None image list."""
        text = "Hello, world!"
        result = create_message(text, None)

        expected = [{"role": "user", "content": "Hello, world!"}]
        assert result == expected

    def test_text_only_with_empty_images(self):
        """Test creating a text-only message with empty image list."""
        text = "Hello, world!"
        result = create_message(text, [])

        expected = [{"role": "user", "content": "Hello, world!"}]
        assert result == expected

    def test_single_image_message(self):
        """Test creating a message with a single image."""
        text = "What's in this image?"
        image_urls = ["https://example.com/image.jpg"]
        result = create_message(text, image_urls)

        expected = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"}
                    },
                    {"type": "text", "text": "What's in this image?"}
                ]
            }
        ]
        assert result == expected

    def test_multiple_images_message(self):
        """Test creating a message with multiple images."""
        text = "Compare these images"
        image_urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg"
        ]
        result = create_message(text, image_urls)

        expected = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image1.jpg"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image2.jpg"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image3.jpg"}
                    },
                    {"type": "text", "text": "Compare these images"}
                ]
            }
        ]
        assert result == expected

    def test_message_structure_order(self):
        """Test that images come before text in the content structure."""
        text = "Analyze this"
        image_urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
        result = create_message(text, image_urls)

        content = result[0]["content"]

        # First two items should be images
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"

        # Last item should be text
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "Analyze this"

    def test_empty_text(self):
        """Test creating a message with empty text."""
        text = ""
        image_urls = ["https://example.com/image.jpg"]
        result = create_message(text, image_urls)

        # Should still create valid structure
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1]["text"] == ""

    def test_single_empty_url(self):
        """Test handling of empty URL string."""
        text = "Test"
        image_urls = [""]
        result = create_message(text, image_urls)

        # Should still create valid structure
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["image_url"]["url"] == ""

    def test_mixed_empty_and_valid_urls(self):
        """Test handling of mix of empty and valid URLs."""
        text = "Test"
        image_urls = ["https://example.com/valid.jpg", "", "https://example.com/valid2.jpg"]
        result = create_message(text, image_urls)

        content = result[0]["content"]
        assert len(content) == 4  # 3 images + 1 text
        assert content[0]["image_url"]["url"] == "https://example.com/valid.jpg"
        assert content[1]["image_url"]["url"] == ""
        assert content[2]["image_url"]["url"] == "https://example.com/valid2.jpg"
        assert content[3]["text"] == "Test"

    def test_very_long_text(self):
        """Test handling of very long text."""
        text = "A" * 10000
        result = create_message(text)

        assert result[0]["content"] == text

    def test_special_characters_in_text(self):
        """Test handling of special characters in text."""
        text = "Hello! 🎉 This has émojis and spëcial chars: @#$%^&*()"
        result = create_message(text)

        assert result[0]["content"] == text

    def test_special_characters_in_urls(self):
        """Test handling of special characters in URLs."""
        text = "Test"
        image_urls = ["https://example.com/image with spaces.jpg", "https://example.com/émoji🎉.png"]
        result = create_message(text, image_urls)

        content = result[0]["content"]
        assert content[0]["image_url"]["url"] == "https://example.com/image with spaces.jpg"
        assert content[1]["image_url"]["url"] == "https://example.com/émoji🎉.png"


@pytest.mark.unit
class TestDetermineDefaultText:
    """Test cases for determine_default_text function."""

    def test_zero_images(self):
        """Test default text for zero images."""
        result = determine_default_text(0)
        assert result == "What's in this image?"

    def test_single_image(self):
        """Test default text for single image."""
        result = determine_default_text(1)
        assert result == "What's in this image?"

    def test_two_images(self):
        """Test default text for two images."""
        result = determine_default_text(2)
        assert result == "I have 2 images. Please describe each image."

    def test_multiple_images(self):
        """Test default text for multiple images."""
        for count in [3, 5, 10, 100]:
            result = determine_default_text(count)
            expected = f"I have {count} images. Please describe each image."
            assert result == expected

    def test_negative_count(self):
        """Test handling of negative image count."""
        result = determine_default_text(-1)
        # Should still work with the logic, though this is an edge case
        assert "I have -1 images" in result

    def test_large_count(self):
        """Test handling of very large image count."""
        result = determine_default_text(1000000)
        assert result == "I have 1000000 images. Please describe each image."


@pytest.mark.unit
class TestMessageIntegration:
    """Integration tests combining message creation functions."""

    def test_create_message_with_determined_text_single(self):
        """Test creating message using determined default text for single image."""
        image_urls = ["https://example.com/image.jpg"]
        text = determine_default_text(len(image_urls))
        result = create_message(text, image_urls)

        expected_text = "What's in this image?"
        assert result[0]["content"][1]["text"] == expected_text

    def test_create_message_with_determined_text_multiple(self):
        """Test creating message using determined default text for multiple images."""
        image_urls = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]
        text = determine_default_text(len(image_urls))
        result = create_message(text, image_urls)

        expected_text = "I have 2 images. Please describe each image."
        assert result[0]["content"][2]["text"] == expected_text

    def test_api_format_compliance(self):
        """Test that created messages comply with expected API format."""
        # Test format from user's example
        text = ("I have two very different images. They are not related at all. "
               "Please describe the first image in one sentence, and then describe the second image in another sentence.")
        image_urls = [
            "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true",
            "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
        ]

        result = create_message(text, image_urls)

        # Verify structure matches expected API format
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 3  # 2 images + 1 text

        # Verify image order and structure
        content = result[0]["content"]
        for i, url in enumerate(image_urls):
            assert content[i]["type"] == "image_url"
            assert content[i]["image_url"]["url"] == url

        # Verify text is last
        assert content[2]["type"] == "text"
        assert content[2]["text"] == text
