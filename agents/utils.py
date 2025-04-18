# 🛠️ utils.py – Shared Utility Functions

def safe_truncate(text: str, max_tokens: int = 8000) -> str:
    """Truncates a string to a safe token limit for LLMs."""
    return text[:max_tokens]

# Add additional utilities as needed
# Placeholder for shared utility functions
