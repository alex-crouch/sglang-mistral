"""Message creation utilities for SGLang Mistral client."""

from typing import Any, Dict, List, Optional


def create_message(text: str, image_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Create a message structure for the SGLang API request.

    Args:
        text: The text content of the message
        image_urls: Optional list of image URLs to include

    Returns:
        List containing a single message dict formatted for the API

    Examples:
        >>> create_message("Hello world")
        [{'role': 'user', 'content': 'Hello world'}]

        >>> create_message("What's this?", ["https://example.com/img.jpg"])
        [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'https://example.com/img.jpg'}},
            {'type': 'text', 'text': "What's this?"}
        ]}]
    """
    content = []

    # Add images first if provided
    if image_urls:
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

    # Add text
    content.append({"type": "text", "text": text})

    # If only text (no images), use simple string format
    if not image_urls:
        return [{"role": "user", "content": text}]

    # Use structured content for text + images
    return [{"role": "user", "content": content}]


def determine_default_text(image_count: int) -> str:
    """
    Determine appropriate default text based on number of images.

    Args:
        image_count: Number of images in the request

    Returns:
        Appropriate default text prompt
    """
    if image_count == 0:
        return "What's in this image?"
    elif image_count == 1:
        return "What's in this image?"
    else:
        return f"I have {image_count} images. Please describe each image."
