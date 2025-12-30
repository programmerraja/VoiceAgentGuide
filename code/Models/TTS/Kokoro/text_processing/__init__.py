"""Text processing pipeline."""

from .normalizer import normalize_text
from .phonemizer import phonemize
from .text_processor import process_text_chunk, smart_split
from .vocabulary import tokenize


def process_text(text: str) -> list[int]:
    """Process text into token IDs (for backward compatibility)."""
    return process_text_chunk(text)


__all__ = [
    "normalize_text",
    "phonemize",
    "tokenize",
    "process_text",
    "process_text_chunk",
    "smart_split",
]
