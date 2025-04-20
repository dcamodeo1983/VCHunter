import os
import numpy as np
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

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
            logging.error(f"Summarization failed: {e}")
            return ""

class EmbedderAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [np.array(e.embedding) for e in response.data]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return []

class GapAnalysisAgent:
    def detect(self, founder_vector, cluster_vectors, labels):
        from sklearn.metrics.pairwise import cosine_similarity

        if founder_vector is None or cluster_vectors is None or len(cluster_vectors) == 0:
            return []

        similarities = cosine_similarity([founder_vector], cluster_vectors)[0]
        sorted_indices = np.argsort(similarities)

        gaps = [{
            "cluster": labels[i],
            "similarity": round(float(similarities[i]), 3)
        } for i in sorted_indices[:3]]

        return gaps

class FounderMatchAgent:
    def match(self, founder_vec, vc_vecs, vc_names, cluster_map):
        from sklearn.metrics.pairwise import cosine_similarity

        scores = cosine_similarity([founder_vec], vc_vecs)[0]
        ranked = sorted(zip(vc_names, scores), key=lambda x: x[1], reverse=True)

        matches = []
        for name, score in ranked[:10]:
            matches.append({
                "vc": name,
                "cluster": cluster_map.get(name, "Unknown"),
                "score": round(float(score), 3)
            })

        return matches

class ChatbotAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def create(self, vc_summaries: list[str], founder_text: str):
        try:
            summary_text = "\n\n".join(vc_summaries)
            prompt = (
                "You're an AI investment advisor. Use the following summaries of VC firms and a founder profile "
                "to help guide the founder on which firms to approach, potential strategies, and questions to expect.\n\n"
                f"Founder:\n{founder_text}\n\nVC Firms:\n{summary_text}"
            )

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Chatbot generation failed: {e}")
            return ""
