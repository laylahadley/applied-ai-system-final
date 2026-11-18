"""Small wrapper around the Google Gemini API.

If there is no API key, the package is missing, or the call fails, every
function returns None so the rest of the system can fall back to offline mode.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("ai_dj")

DEFAULT_MODEL = "gemini-2.5-flash"


def _get_api_key() -> Optional[str]:
    """Read GEMINI_API_KEY from the environment or a .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def llm_available() -> bool:
    """True if an API key is configured."""
    return _get_api_key() is not None


def ask_llm(prompt: str) -> Optional[str]:
    """Send a prompt to Gemini and return the text reply, or None on any problem."""
    key = _get_api_key()
    if key is None:
        logger.info("No GEMINI_API_KEY found; using offline mode.")
        return None

    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai is not installed; using offline mode.")
        return None

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = (response.text or "").strip()
        logger.info("LLM call succeeded (model=%s, %d chars).", model, len(text))
        return text or None
    except Exception as exc:  # network errors, bad key, quota, etc.
        logger.warning("LLM call failed (%s); using offline mode.", exc)
        return None