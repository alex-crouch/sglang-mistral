"""SGLang Mistral client for making requests to the inference server."""

import os
import sys
from typing import Any, Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional, continue without it
    pass


class SGLangMistralClient:
    """Client for interacting with SGLang Mistral inference server."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the SGLang Mistral client.

        Args:
            host: Server host address (defaults to SGLANG_HOST env var or localhost)
            port: Server port (defaults to SGLANG_PORT env var or 30000)
            model: Model name (defaults to OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym)
        """
        self.host = host or os.getenv("SGLANG_HOST", "localhost")
        self.port = port or int(os.getenv("SGLANG_PORT", "30000"))
        self.model = model or "OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
        self.base_url = f"http://{self.host}:{self.port}"

    def make_request(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 300,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Make a request to the SGLang server.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters to pass to the API

        Returns:
            Response from the server as a dictionary

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/v1/chat/completions"

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}", file=sys.stderr)
            raise

    def make_request_raw(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 300,
        **kwargs: Any
    ) -> str:
        """
        Make a request to the SGLang server and return raw text response.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters to pass to the API

        Returns:
            Raw response text from the server

        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.base_url}/v1/chat/completions"

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}", file=sys.stderr)
            raise

    def health_check(self) -> bool:
        """
        Check if the SGLang server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def __repr__(self) -> str:
        """Return string representation of the client."""
        return f"SGLangMistralClient(host='{self.host}', port={self.port}, model='{self.model}')"
