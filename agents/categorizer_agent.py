import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
from collections import defaultdict
import logging

class CategorizerAgentV2:
    def __init__(self, api_key: str, method: str = "kmeans", n_clusters: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.method = method
        self.default_n_clusters = n_clusters

    def cluster_embeddings(self, embeddings: np.ndarray, vc_ids: list) -> dict:
        labels_map = defaultdict(list)

        if len(vc_ids) < 2:
            logging.warning("Too few VCs to cluster — skipping clustering.")
            return {0: vc_ids}

        n_clusters = min(self.default_n_clusters, len(vc_ids))
        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings)

        for idx, label in enumerate(labels):
            labels_map[label].append(vc_ids[idx])
        return labels_map

    def explain_cluster(self, vc_urls_in_cluster: list, summaries: dict) -> str:
        joined_summaries = "\n\n".join(
            [f"{url}:\n{summaries[url]}" for url in vc_urls_in_cluster if url in summaries]
        )

        prompt = f"""
These VC firms have been grouped together based on similar investment behavior.

Please:
1. Suggest a category name
2. Summarize their common strategy or themes
3. Identify shared founder preferences

Data:
{joined_summaries}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"LLM explanation error: {e}")
            return "(Explanation failed)"

    def categorize(self, embeddings: np.ndarray, vc_ids: list, summaries: dict) -> list:
        if len(vc_ids) < 2:
            logging.warning("Fewer than 2 VC profiles available — categorization skipped.")
            return [{"cluster_id": 0, "members": vc_ids, "description": "Single cluster: All available VCs"}]

        clusters = self.cluster_embeddings(embeddings, vc_ids)
        result = []

        for cluster_id, members in clusters.items():
            description = self.explain_cluster(members, summaries)
            result.append({
                "cluster_id": cluster_id,
                "members": members,
                "description": description
            })

        return result
