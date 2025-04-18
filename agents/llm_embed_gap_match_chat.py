# 🧠 LLMSummarizerAgentV2 – Robust Structured Summarization
import openai
import json
import logging

class LLMSummarizerAgentV2:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def summarize(self, vc_text: str, portfolio_text: str = "") -> dict:
        prompt = f"""
Summarize the following VC firm and their portfolio company content in JSON format with the following keys:
- thesis (string)
- sectors (list)
- stage (string)
- region (string)
- tone (string)
- founder_fit (string)
- observed_behavior (string) — what does their portfolio suggest about how they actually invest?
- portfolio_keywords (list)

VC Website Content:
{vc_text}

Portfolio Company Content:
{portfolio_text}
"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            return json.loads(content.strip())
        except json.JSONDecodeError:
            logging.warning("Initial response was not valid JSON. Returning raw output.")
            return {"raw_summary": content.strip()}
        except Exception as e:
            logging.error(f"LLM summary error: {e}")
            return {}



# 🔎 EmbedderAgentV2 – With diagnostics and FAISS index optional
import numpy as np
import faiss

class EmbedderAgentV2:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def embed(self, texts: list) -> np.ndarray:
        try:
            response = openai.Embedding.create(
                input=texts, model="text-embedding-ada-002"
            )
            return np.array([x['embedding'] for x in response['data']])
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            return np.zeros((len(texts), 1536))

    def build_faiss_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index



# 🔍 GapAnalysisAgentV2 – Smart Opportunity Scanner
from sklearn.metrics.pairwise import cosine_similarity

class GapAnalysisAgentV2:
    def detect(self, founder_vec: np.ndarray, cluster_centroids: np.ndarray, labels: list) -> list:
        sims = cosine_similarity(founder_vec.reshape(1, -1), cluster_centroids).flatten()
        ranked = np.argsort(sims)[:3]  # lowest = underserved
        return [
            {
                "category": labels[i],
                "score": float(sims[i]),
                "insight": f"Opportunity: Low similarity to cluster '{labels[i]}'. This may indicate white space."
            }
            for i in ranked
        ]



# 🎯 FounderMatchAgentV2 – Contextual Matching
class FounderMatchAgentV2:
    def match(self, founder_embedding: np.ndarray, vc_embeddings: np.ndarray, vc_names: list, category_map: dict) -> list:
        sims = cosine_similarity(founder_embedding.reshape(1, -1), vc_embeddings).flatten()
        top_k = sims.argsort()[-5:][::-1]
        results = []
        for i in top_k:
            results.append({
                "vc": vc_names[i],
                "score": float(sims[i]),
                "category": category_map.get(vc_names[i], "Uncategorized"),
                "why": f"Match based on embedding similarity and shared cluster: {category_map.get(vc_names[i], 'N/A')}"
            })
        return results



# 💬 ChatbotAgentV2 – Grounded Semantic Q&A
class ChatbotAgentV2:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def respond(self, question: str, context_docs: list) -> str:
        context = "\n\n".join(context_docs[:5])  # top-k docs
        messages = [
            {"role": "system", "content": "You are a helpful VC landscape expert. Use the context below to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Chatbot failed: {e}")
            return "Sorry, I couldn't retrieve an answer."

