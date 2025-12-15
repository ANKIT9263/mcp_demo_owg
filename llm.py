import os
import requests
from typing import List, Dict, Any, Optional
import json
from dotenv import load_dotenv

load_dotenv()


class ChatMMC:
    """
    Custom LLM class similar to ChatOpenAI that integrates with MMC's internal API.
    Can be used as a drop-in replacement for ChatOpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the ChatMMC client.

        Args:
            api_key: API key for authentication (defaults to env vars)
            model: Model deployment name (defaults to env vars)
            base_url: Base URL for the API (defaults to env vars)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        # Load from environment variables with fallbacks
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ORG_LLM_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("ORG_LLM_MODEL") or "mmc-tech-gpt-41-1m-2025-04-14"
        self.base_url = base_url or os.getenv("ORG_LLM_BASE_URL") or "https://stg1.mmc-dallas-int-non-prod-ingress.mgti.mmc.com/coreapi/openai/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = f"{self.base_url}/deployments/{self.model}/chat/completions"
        self.kwargs = kwargs

        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or via OPENAI_API_KEY/ORG_LLM_API_KEY environment variable")

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Invoke the LLM with a list of messages.
        Compatible with langchain's ChatOpenAI interface.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters to override instance settings

        Returns:
            The content of the assistant's response
        """
        # Handle langchain message objects
        formatted_messages = self._format_messages(messages)

        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature)
        }

        # Add optional parameters
        if self.max_tokens or kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens"]:
                payload[key] = value

        # Make API request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        # Parse response
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _format_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        Format messages to ensure they're in the correct format.
        Handles both dict messages and langchain message objects.

        Args:
            messages: List of messages (dicts or langchain objects)

        Returns:
            List of formatted message dictionaries
        """
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(msg)
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                # Langchain message object
                formatted.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                # Alternative langchain format
                role_mapping = {
                    "system": "system",
                    "human": "user",
                    "ai": "assistant",
                    "user": "user",
                    "assistant": "assistant"
                }
                formatted.append({
                    "role": role_mapping.get(msg.type, msg.type),
                    "content": msg.content
                })
            else:
                raise ValueError(f"Unsupported message format: {type(msg)}")

        return formatted

    def predict(self, text: str, **kwargs) -> str:
        """
        Simple prediction interface (langchain compatibility).

        Args:
            text: User message text
            **kwargs: Additional parameters

        Returns:
            The assistant's response
        """
        messages = [{"role": "user", "content": text}]
        return self.invoke(messages, **kwargs)

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Allow the instance to be called directly.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Returns:
            The assistant's response
        """
        return self.invoke(messages, **kwargs)

    def stream(self, messages: List[Dict[str, str]], **kwargs):
        """
        Stream responses from the LLM (if supported by the API).
        Note: This is a placeholder - implement if streaming is supported.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters

        Yields:
            Chunks of the response
        """
        # For now, just return the full response
        # Implement actual streaming if the API supports it
        response = self.invoke(messages, **kwargs)
        yield response


# Example usage
if __name__ == "__main__":
    # Initialize the client
    llm = ChatMMC()

    # Example 1: Using invoke with messages
    messages = [
        {
            "role": "system",
            "content": "You are an AI information extractor for a pitch deck."
        },
        {
            "role": "user",
            "content": "Client has struggled with inconsistent operating margins since 2021 and wants to reduce supply chain costs by 20%"
        }
    ]

    response = llm.invoke(messages)
    print("Response:", response)

    # Example 2: Using predict
    simple_response = llm.predict("What is the capital of France?")
    print("Simple response:", simple_response)

    # Example 3: Direct call
    direct_response = llm(messages)
    print("Direct call response:", direct_response)
