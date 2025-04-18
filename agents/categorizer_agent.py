# 🧠 CategorizerAgentV2 – Merged Advanced Version
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from openai import OpenAI
from collections import defaultdict
import logging

class CategorizerAgentV2:
    def __init__(self, api_key: str, method: str = "ensemble", n_clusters: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.method = method
        self.n_clusters = n_clusters

    def cluster_embeddings(self, embeddings: np.ndarray, vc_ids: list) -> dict:
        labels_map = defaultdict(list)

        if self.method == "kmeans":
            labels = KMeans(n_clusters=self.n_clusters, random_state=42).fit_predict(embeddings)
        elif self.method == "hdbscan":
            labels = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(embeddings)
        elif self.method == "ensemble":
            km = KMeans(n_clusters=self.n_clusters, random_state=42).fit_predict(embeddings)
            hb = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(embeddings)
            labels = [h if h != -1 else km[i] for i, h in enumerate(hb)]
        else:
            raise ValueError("Invalid clustering method.")

        for idx, label in enumerate(labels):
            if label != -1:
                labels_map[label].append(vc_ids[idx])

        return labels_map

    def explain_cluster(self, vc_urls_in_cluster: list, summaries: dict) -> str:
        joined_summaries = "\n\n".join(
            [f"{url}:\n{summaries[url]}" for url in vc_urls_in_cluster if url in summaries]
        )

        prompt = f"""
These VC firms have been grouped together based on similarity in their investment strategy.

For this group, please:
1. Suggest a name
2. Describe common strategies, themes, or founder preferences
3. List the VC firm URLs

Data:
{joined_summaries}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"LLM explanation error: {e}")
            return "(Explanation failed)"

    def categorize(self, embeddings: np.ndarray, vc_ids: list, summaries: dict) -> list:
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

