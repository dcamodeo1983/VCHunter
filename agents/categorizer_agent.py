import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
from collections import defaultdict
import logging

class CategorizerAgentV2:
    def __init__(self, api_key: str, method: str = "kmeans", n_clusters: int = 5):
        self.client = OpenAI(api_key=api_key)
        self.method = method
        self.n_clusters = n_clusters

    def cluster_embeddings(self, embeddings: np.ndarray, vc_ids: list) -> dict:
        labels_map = defaultdict(list)
        labels = KMeans(n_clusters=self.n_clusters, random_state=42).fit_predict(embeddings)

        for idx, label in enumerate(labels):
            labels_map[label].append(vc_ids[idx])
        return labels_map

    def explain_cluster(self, vc_urls_in_cluster: list, summaries: dict) -> str:
        joined_summaries = "\n\n".join(
            [f"{url}:\n{summaries[url]}" for url in vc_urls_in_cluster if url in summaries]
        )

        prompt = f"""
These VC firms have been grouped together based on similar investment strategy.

For this group, please:
1. Suggest a name
2. Describe common strategies, themes, or founder preferences
3. List the VC firm URLs

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
