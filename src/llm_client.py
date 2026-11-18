"""Small wrapper around the Google Gemini API.

If there is no API key, the package is missing, or the call fails, every
function returns None so the rest of the system can fall back to offline mode.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_dj")

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 4
RETRY_WAIT_SECONDS = 3
RETRYABLE_CODES = ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _read_env_file(name: str) -> Optional[str]:
    """Read one setting from the project's .env file (handles Windows encodings)."""
    if not ENV_PATH.exists():
        return None
    raw = ENV_PATH.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None
    for line in text.replace("\x00", "").splitlines():
        line = line.strip()
        if line.startswith(name):
            _, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def _get_api_key() -> Optional[str]:
    """Read GEMINI_API_KEY from the environment, or from the .env file in the project folder."""
    key = os.getenv("GEMINI_API_KEY", "").strip() or _read_env_file("GEMINI_API_KEY")
    if key is None:
        logger.info("No key in environment or %s (file exists: %s).", ENV_PATH, ENV_PATH.exists())
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

    model = os.getenv("GEMINI_MODEL", "").strip() or _read_env_file("GEMINI_MODEL") or DEFAULT_MODEL
    client = genai.Client(api_key=key)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = (response.text or "").strip()
            logger.info("LLM call succeeded (model=%s, attempt %d, %d chars).", model, attempt, len(text))
            return text or None
        except Exception as exc:  # network errors, bad key, quota, overload, etc.
            temporary = any(code in str(exc) for code in RETRYABLE_CODES)
            if temporary and attempt < MAX_ATTEMPTS:
                wait = RETRY_WAIT_SECONDS * attempt
                logger.warning("LLM attempt %d failed (temporary: %s); retrying in %ds.",
                               attempt, str(exc)[:80], wait)
                time.sleep(wait)
                continue
            logger.warning("LLM call failed after %d attempt(s) (%s); using offline mode.", attempt, exc)
            return None
    return None