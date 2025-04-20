import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 🧠 Summarizer Agent
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
            return ""

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


# 📐 Embedder Agent
class EmbedderAgent:
    def __init__(self, api_key):
        openai.api_key = api_key

    def embed(self, texts: list) -> list:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [r.embedding for r in response.data]
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            return []


# 🧠 Gap Analysis Agent
class GapAnalysisAgent:
    def detect(self, founder_vec, centroids, labels):
        try:
            sims = cosine_similarity(founder_vec.reshape(1, -1), centroids).flatten()
            best = np.argmax(sims)
            gaps = []
            for i, sim in enumerate(sims):
                if sim < 0.6:
                    gaps.append({
                        "label": labels[i],
                        "reason": f"Founder profile is not aligned with cluster {labels[i]} (similarity: {sim:.2f})"
                    })
            return gaps
        except Exception as e:
            logging.error(f"Gap detection failed: {e}")
            return []


# 🎯 Matching Agent
class FounderMatchAgent:
    def match(self, founder_vec, vc_vecs, vc_names, vc_to_cluster):
        sims = cosine_similarity(founder_vec.reshape(1, -1), vc_vecs).flatten()
        results = []
        for i, score in enumerate(sims):
            results.append({
                "vc": vc_names[i],
                "score": float(score),
                "cluster": vc_to_cluster[vc_names[i]]
            })
        sorted_results = sorted(results, key=lambda x: -x["score"])
        return sorted_results[:5]


# 💬 Chatbot Agent
class ChatbotAgent:
    def __init__(self, api_key):
        self.api_key = api_key

    def create(self, vc_summaries, founder_text):
        context = "\n\n".join(vc_summaries)
        return SimpleChatResponder(context, self.api_key)


class SimpleChatResponder:
    def __init__(self, context, api_key):
        self.context = context
        openai.api_key = api_key

    def ask(self, question):
        try:
            messages = [
                {"role": "system", "content": "You are an expert VC analyst. Use the context to answer the user's question clearly."},
                {"role": "user", "content": f"Context:\n{self.context}\n\nQuestion: {question}"}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Chatbot failed: {e}")
            return "⚠️ Error answering your question."
