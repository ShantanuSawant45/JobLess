import httpx

from app.config import get_settings

# Gemini API endpoint for the flash model (fast and cheap, good for RAG)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)


def generate(prompt: str) -> str:
    """
    Send a prompt to the Gemini API and return the generated text.

    Args:
        prompt: The full formatted prompt (system instructions + context + question).

    Returns:
        The LLM's text response as a plain string.

    Raises:
        httpx.HTTPStatusError: If the Gemini API returns a non-2xx status.
    """
    api_key = get_settings().llm_api_key.get_secret_value()

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = httpx.post(
        GEMINI_API_URL,
        params={"key": api_key},
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()

    data = response.json()

    # Extract the text from Gemini's response structure:
    # data["candidates"][0]["content"]["parts"][0]["text"]
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response structure: {data}") from e
