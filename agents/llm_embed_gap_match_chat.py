# ✅ TEMP: Basic LLMSummarizerAgent for import testing

import openai
import logging

class LLMSummarizerAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def summarize(self, site_text: str, portfolio_text: str) -> str:
        try:
            prompt = f"""
You are a VC analyst. Summarize the investment behavior of this firm using site and portfolio data.

Firm Website:
{site_text}

Portfolio:
{portfolio_text}

Summarize with emphasis on:
1. Investment thesis
2. Types of founders they prefer
3. Example markets
4. Implicit themes
"""
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return "⚠️ Error summarizing VC."

    def summarize_founder(self, text: str) -> str:
        try:
            prompt = f"""
Summarize the following startup description. Extract key attributes like domain, market, traction, and product.

Text:
{text}
"""
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Founder summarization failed: {e}")
            return "⚠️ Error summarizing founder document."
