# utils.py — Common utility functions

import re
import unicodedata
from typing import Set


def clean_text(text: str) -> str:
    """
    Normalize and clean up text for more consistent processing.
    
    - Removes extra whitespace.
    - Converts unicode characters to ASCII.
    - Strips non-ASCII characters.
    
    Args:
        text (str): The input text.
    
    Returns:
        str: Cleaned, normalized text.
    """
    if not text:
        return ""

    # Unicode normalization and cleanup
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1 (Set[str]): First set.
        set2 (Set[str]): Second set.
    
    Returns:
        float: Jaccard similarity coefficient.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0
