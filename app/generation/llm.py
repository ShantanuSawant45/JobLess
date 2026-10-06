import httpx
import time

from app.config import get_settings

# Gemini API endpoint for the flash model (fast and cheap, good for RAG)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def generate(prompt: str, max_retries: int = 5) -> str:
    """
    Send a prompt to the Gemini API and return the generated text.
    Includes exponential backoff for rate limiting (HTTP 429).

    Args:
        prompt: The full formatted prompt (system instructions + context + question).
        max_retries: Maximum number of times to retry on 429 errors.

    Returns:
        The LLM's text response as a plain string.

    Raises:
        httpx.HTTPStatusError: If the Gemini API returns a non-2xx status (other than 429 after retries).
    """
    api_key = get_settings().llm_api_key.get_secret_value()

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    delay = 2.0
    for attempt in range(max_retries):
        response = httpx.post(
            GEMINI_API_URL,
            params={"key": api_key},
            json=payload,
            timeout=30.0,
        )
        
        if response.status_code == 429:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # exponential backoff
                continue
            else:
                response.raise_for_status()
                
        response.raise_for_status()
        break

    data = response.json()
    
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response structure: {data}") from e
