import os
import numpy as np
import logging
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)

# 🔍 Summarization
class LLMSummarizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize_founder(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Summarize the startup concept in the following founder description:\n\n{text}"
                }]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Founder summarization failed: {e}")
            return ""

    def summarize(self, site_text: str, portfolio_text: str) -> str:
        try:
            combined = f"{site_text}\n\nPortfolio:\n{portfolio_text}"
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Summarize this VC firm based on their website and portfolio:\n\n{combined}"
                }]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"VC summarization failed: {e
