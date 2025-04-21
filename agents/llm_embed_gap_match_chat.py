from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging

class LLMSummarizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def summarize(self, site_text, portfolio_text):
        try:
            content = f"Summarize the VC firm based on their website:\n\n{site_text[:3000]}\n\nPortfolio:\n{portfolio_text[:3000]}"
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": content}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Summarization failed: {e}")
            return ""

    def summarize_founder(self, text):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Summarize this founder's idea:\n\n{text[:4000]}"}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Founder summarization failed: {e}")
            return ""


class EmbedderAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts):
        try:
            if not texts:
                return []
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [e.embedding for e in response.data]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return []


class FounderMatchAgent:
    def match(self, founder_vec, vc_vecs, vc_names, cluster_map):
        try:
            scores = cosine_similarity([founder_vec], vc_vecs)[0]
            ranked = sorted(zip(vc_names, scores), key=lambda x: x[1], reverse=True)
            return [{"vc": name, "score": round(score, 3), "cluster": cluster_map.get(name, "N/A")} for name, score in ranked]
        except Exception as e:
            logging.error(f"Founder matching failed: {e}")
            return []


class ChatbotAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def create(self, vc_summaries, founder_summary):
        try:
            content = f"Here are VC summaries:\n{vc_summaries}\n\nAnd the founder summary:\n{founder_summary}"
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": content}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Chatbot generation failed: {e}")
            return "Chatbot could not generate a response."


class GapAnalysisAgent:
    def detect(self, founder_vec, embeddings, cluster_ids):
        try:
            similarities = cosine_similarity([founder_vec], embeddings)[0]
            return [{"cluster": cid, "similarity": round(similarities[i], 3)} for i, cid in enumerate(cluster_ids)]
        except Exception as e:
            logging.error(f"Gap detection failed: {e}")
            return []
