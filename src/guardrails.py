"""Reliability guardrails for the AI DJ.

1. validate_request  - checks the user's text before anything else runs.
2. validate_profile  - cleans the parsed taste profile and flags problems.
3. check_dj_note     - verifies the AI's written answer only mentions real,
                       retrieved songs (catches hallucinated songs).
4. confidence_score  - turns the top match score into a 0-1 confidence.
"""

import re
from typing import Dict, Iterable, List, Tuple

MAX_REQUEST_CHARS = 300
DEFAULT_ENERGY = 0.5

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "ignore the above",
    "system prompt",
    "you are now",
]


class GuardrailError(ValueError):
    """Raised when a request is rejected by an input guardrail."""


def validate_request(text) -> str:
    """Return a cleaned request string, or raise GuardrailError with a friendly reason."""
    if not isinstance(text, str):
        raise GuardrailError("Request must be text.")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise GuardrailError("Request is empty. Try something like 'chill music to study to'.")
    if len(cleaned) > MAX_REQUEST_CHARS:
        raise GuardrailError(f"Request is too long ({len(cleaned)} chars, max {MAX_REQUEST_CHARS}).")
    if not re.search(r"[a-zA-Z]", cleaned):
        raise GuardrailError("Request has no words in it. Describe the music you want.")
    lowered = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lowered:
            raise GuardrailError("Request looks like a prompt-injection attempt and was blocked.")
    return cleaned


def validate_profile(
    prefs: Dict, known_genres: Iterable[str], known_moods: Iterable[str]
) -> Tuple[Dict, List[str]]:
    """Clamp energy to 0-1, normalize text, and return (clean_prefs, warnings)."""
    warnings: List[str] = []
    known_genres = {g.lower() for g in known_genres}
    known_moods = {m.lower() for m in known_moods}

    genre = str(prefs.get("favorite_genre") or "").strip().lower()
    mood = str(prefs.get("favorite_mood") or "").strip().lower()

    try:
        energy = float(prefs.get("target_energy"))
    except (TypeError, ValueError):
        energy = DEFAULT_ENERGY
        warnings.append(f"Energy was missing or invalid; using default {DEFAULT_ENERGY}.")
    if energy < 0.0 or energy > 1.0:
        clamped = min(1.0, max(0.0, energy))
        warnings.append(f"Energy {energy} was out of range; clamped to {clamped}.")
        energy = clamped

    if not genre:
        warnings.append("No genre detected; ranking uses mood and energy only.")
    elif genre not in known_genres:
        warnings.append(f"Genre '{genre}' is not in the catalog; ranking uses mood and energy only.")

    if not mood:
        warnings.append("No mood detected; ranking uses genre and energy only.")
    elif mood not in known_moods:
        warnings.append(f"Mood '{mood}' is not in the catalog; ranking uses genre and energy only.")

    clean = {"favorite_genre": genre, "favorite_mood": mood, "target_energy": round(energy, 2)}
    return clean, warnings


def _mentions(text: str, title: str) -> bool:
    """True if the title appears in the text as whole words (case-insensitive)."""
    pattern = r"(?<!\w)" + re.escape(title.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def check_dj_note(
    note: str, recommended_titles: List[str], catalog_titles: List[str]
) -> Tuple[bool, List[str]]:
    """Check an AI-written note. Returns (passed, list_of_problems)."""
    problems: List[str] = []
    if not note or not note.strip():
        return False, ["Note is empty."]

    recommended_lower = {t.lower() for t in recommended_titles}
    catalog_lower = {t.lower() for t in catalog_titles}

    missing = [t for t in recommended_titles if not _mentions(note, t)]
    if missing:
        problems.append(f"Note skipped recommended songs: {missing}")

    off_list = [t for t in catalog_titles if t.lower() not in recommended_lower and _mentions(note, t)]
    if off_list:
        problems.append(f"Note mentions catalog songs that were not recommended: {off_list}")

    quoted = re.findall(r'"([^"]{2,60})"', note)
    invented = [q for q in quoted if q.lower() not in catalog_lower]
    if invented:
        problems.append(f"Note mentions songs that do not exist in the catalog: {invented}")

    return (len(problems) == 0), problems


def confidence_score(top_score: float, max_score: float) -> Tuple[float, str]:
    """Return (confidence 0-1, label) based on how close the best match is to a perfect score."""
    if max_score <= 0:
        return 0.0, "low"
    value = max(0.0, min(1.0, top_score / max_score))
    if value >= 0.75:
        label = "high"
    elif value >= 0.5:
        label = "medium"
    else:
        label = "low"
    return round(value, 2), label