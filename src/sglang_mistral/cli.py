#!/usr/bin/env python3
"""
Command-line interface for SGLang Mistral client.

This module provides the main CLI entry point for interacting with
SGLang Mistral inference server.
"""

import argparse
import sys
from typing import List, Optional

from .client import SGLangMistralClient
from .message import create_message, determine_default_text


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Interact with SGLang Mistral model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "Hello, how are you?"
  %(prog)s --image-url "https://example.com/image.jpg"
  %(prog)s --text "What's in these images?" --image-url "https://example.com/img1.jpg" --image-url "https://example.com/img2.jpg"
  %(prog)s --text "What's in these images?" --image-url "https://example.com/img1.jpg" "https://example.com/img2.jpg"
  %(prog)s --host 192.168.1.100 --port 8080 --text "Hello"
        """
    )

    # Server configuration
    parser.add_argument(
        "--host", "--addr",
        help="Server host address (default: localhost or SGLANG_HOST env var)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: 30000 or SGLANG_PORT env var)"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        help="Model name (default: OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum tokens in response (default: 300)"
    )

    # Input configuration
    parser.add_argument(
        "--text",
        help="Text message to send (optional, defaults based on context)"
    )
    parser.add_argument(
        "--image-url",
        nargs='*',
        action="append",
        help="Image URL to analyze (can be used multiple times, or provide multiple URLs after single flag)"
    )

    # Output options
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw response text instead of parsed JSON"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output formatted JSON response"
    )

    return parser


def determine_text_and_images(
    text_arg: Optional[str],
    image_urls_arg: Optional[List[str]]
) -> tuple[str, Optional[List[str]]]:
    """
    Determine the final text message and image URLs based on arguments.

    Args:
        text_arg: Text provided via --text argument
        image_urls_arg: Image URLs provided via --image-url arguments

    Returns:
        Tuple of (text_message, image_urls)
    """
    # Determine text message
    if text_arg is not None:
        text_message = text_arg
        image_urls = image_urls_arg
    elif image_urls_arg:
        # Default text when images are provided but no text specified
        text_message = determine_default_text(len(image_urls_arg))
        image_urls = image_urls_arg
    else:
        # No images and no text provided, use default
        text_message = determine_default_text(1)
        image_urls = ["https://github.com/alex-crouch/resources/blob/main/zsh-diskfree/images/servingsuggestion.gif?raw=true"]

    return text_message, image_urls


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Create client
        client = SGLangMistralClient(
            host=args.host,
            port=args.port,
            model=args.model
        )

        # Flatten image URLs from nested lists (due to nargs='*' + action='append')
        flattened_image_urls = None
        if args.image_url:
            flattened_image_urls = []
            for url_group in args.image_url:
                flattened_image_urls.extend(url_group)

        # Determine text and images
        text_message, image_urls = determine_text_and_images(
            args.text,
            flattened_image_urls
        )

        # Create messages
        messages = create_message(text_message, image_urls)

        # Make request
        if args.raw:
            response = client.make_request_raw(
                messages=messages,
                max_tokens=args.max_tokens
            )
            print(response)
        else:
            response = client.make_request(
                messages=messages,
                max_tokens=args.max_tokens
            )

            if args.json:
                import json
                print(json.dumps(response, indent=2))
            else:
                print(response)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
