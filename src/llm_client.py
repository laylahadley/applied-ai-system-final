"""Small wrapper around the Google Gemini API.

If there is no API key, the package is missing, or the call fails, every
function returns None so the rest of the system can fall back to offline mode.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_dj")
_gemini_unreachable = False  # set True after a timeout so later calls skip Gemini

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2
CALL_TIMEOUT_SECONDS = 15   # give up on any single call after this long
TOTAL_BUDGET_SECONDS = 25   # never spend longer than this on one request, retries included
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

    global _gemini_unreachable
    if _gemini_unreachable:
        logger.info("Skipping Gemini: it timed out earlier in this run; using offline mode.")
        return None

    model = os.getenv("GEMINI_MODEL", "").strip() or _read_env_file("GEMINI_MODEL") or DEFAULT_MODEL
    try:
        client = genai.Client(api_key=key)
    except Exception as exc:
        logger.warning("Could not create Gemini client (%s); using offline mode.", exc)
        return None

    started = time.monotonic()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _call_with_timeout(
                lambda: client.models.generate_content(model=model, contents=prompt),
                CALL_TIMEOUT_SECONDS,
            )
            text = (response.text or "").strip()
            logger.info("LLM call succeeded (model=%s, attempt %d, %d chars).", model, attempt, len(text))
            return text or None
        except TimeoutError:
            _gemini_unreachable = True
            logger.warning("LLM call timed out after %ds; skipping Gemini for the rest of this run.",
                           CALL_TIMEOUT_SECONDS)
            return None
        except Exception as exc:  # network errors, bad key, quota, overload, etc.
            temporary = any(code in str(exc) for code in RETRYABLE_CODES)
            wait = RETRY_WAIT_SECONDS * attempt
            within_budget = (time.monotonic() - started + wait) < TOTAL_BUDGET_SECONDS
            if temporary and attempt < MAX_ATTEMPTS and within_budget:
                logger.warning("LLM attempt %d failed (temporary: %s); retrying in %ds.",
                               attempt, str(exc)[:80], wait)
                time.sleep(wait)
                continue
            logger.warning("LLM call failed after %d attempt(s) (%s); using offline mode.", attempt, exc)
            return None
    return None


def _call_with_timeout(func, seconds: float):
    """Run func() in a background thread; raise TimeoutError if it takes too long.

    A daemon thread is used so a stuck network call can never keep the program open.
    """
    box: dict = {}

    def runner():
        try:
            box["value"] = func()
        except Exception as exc:  # pass the error back to the caller
            box["error"] = exc

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"no reply within {seconds}s")
    if "error" in box:
        raise box["error"]
    return box["value"]