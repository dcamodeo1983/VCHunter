from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging


# 🔡 Embedding Agent
class EmbedderAgent:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts):
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return []


# 🧠 Summarization Agent
class LLMSummarizerAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, site_text: str, portfolio_text: str):
        prompt = f"""
You are an expert in venture capital intelligence.

Analyze the following VC firm information and return:
1. The stated thesis (quote if available)
2. Latent or implied themes
3. Preferred founder type
4. Any strategic contradictions
5. 2-3 sentence profile summary

== Website Content ==
{site_text}

== Portfolio Description ==
{portfolio_text}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return "(Summary unavailable)"


# 🔍 Matching Agent
class FounderMatchAgent:
    def __init__(self):
        pass

    def match(self, founder_embedding, vc_embeddings, vc_names, vc_to_cluster):
        if founder_embedding is None or len(vc_embeddings) == 0:
            return [{
                "name": "(No VC data available)",
                "score": 0.0,
                "cluster": None
            }]

        try:
            sims = cosine_similarity(
                founder_embedding.reshape(1, -1),
