# utils.py — Common utility functions

import re
import unicodedata
import logging
from typing import Set

logging.basicConfig(level=logging.INFO)

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

    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    cleaned = text.strip()
    logging.debug(f"Cleaned text length: {len(cleaned)}")
    return cleaned


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
    score = len(intersection) / len(union) if union else 0.0
    logging.debug(f"Jaccard similarity: {score:.4f}")
    return score
