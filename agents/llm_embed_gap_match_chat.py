from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

# 🧠 Summarizer Agent
class LLMSummarizerAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, site_text: str, portfolio_text: str) -> str:
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


# 🔡 Embedding Agent
class EmbedderAgent:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list:
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return []


# 🎯 Match Agent
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
                vc_embeddings
            ).flatten()
        except Exception as e:
            return [{
                "name": "(Similarity failed)",
                "score": 0.0,
                "cluster": None,
                "error": str(e)
            }]

        top_matches = sorted(
            zip(vc_names, sims),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return [{
            "name": name,
            "score": float(score),
            "cluster": vc_to_cluster.get(name, None)
        } for name, score in top_matches]


# 🌌 Gap Analysis Agent
class GapAnalysisAgent:
    def __init__(self):
        pass

    def detect(self, founder_embedding, centroids, labels):
        if founder_embedding is None or len(centroids) == 0:
            return []

        try:
            sims = cosine_similarity(founder_embedding.reshape(1, -1), centroids).flatten()
        except Exception as e:
            return [{"category": "Error", "score": 0.0, "insight": str(e)}]

        ranked = sorted(zip(labels, sims), key=lambda x: x[1])
        insights = [{
            "category": label,
            "score": float(score),
            "insight": f"This category is underrepresented in your VC match profile (score: {score:.2f})"
        } for label, score in ranked[:3]]

        return insights


# 🤖 Chatbot Agent (Optional interactive mode)
class ChatbotAgent:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Chatbot failed: {e}")
            return "(Chat unavailable)"
