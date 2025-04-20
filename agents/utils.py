# utils.py — Common utility functions

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalize and clean up text for more consistent processing.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0
