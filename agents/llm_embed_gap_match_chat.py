import openai
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LLMSummarizerAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def summarize_founder(self, text):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"Summarize this startup and describe its industry, market, and traction: {text}"
                }],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Founder summarization failed: {e}")
            return None

    def summarize(self, site_text, portfolio_text):
        try:
            prompt = (
                "You are an analyst summarizing a VC firm. Use the website content and portfolio info to describe:\n"
                "- Investment themes\n"
                "- Notable companies\n"
                "- Stage and geography focus\n"
                f"Website:\n{site_text}\n\nPortfolio:\n{portfolio_text}"
            )
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return None

class EmbedderAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def embed(self, texts):
        try:
            response = openai.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            return [np.array(e.embedding) for e in response.data]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return []

class GapAnalysisAgent:
    def detect(self, founder_vec, cluster_centroids, labels):
        if len(cluster_centroids) == 0:
            return []
        sims = cosine_similarity(founder_vec.reshape(1, -1), cluster_centroids).flatten()
        return [{"cluster": labels[i], "distance": round(1 - s, 4)} for i, s in enumerate(sims)]

class FounderMatchAgent:
    def match(self, founder_vec, vc_vecs, vc_names, vc_to_cluster):
        sims = cosine_similarity(founder_vec.reshape(1, -1), vc_vecs).flatten()
        ranked = sorted(zip(vc_names, sims), key=lambda x: x[1], reverse=True)
        return [{"vc": name, "similarity": round(score, 3), "cluster": vc_to_cluster.get(name, "")}
                for name, score in ranked[:10]]

class ChatbotAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def create(self, summaries, founder_text):
        return "🧠 Chat feature placeholder (chatbot not yet implemented)"
