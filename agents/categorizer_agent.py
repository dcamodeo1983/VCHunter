import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import openai
import logging

class CategorizerAgent:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters

    def categorize(self, embeddings, vc_names, summaries_dict):
        try:
            if len(embeddings) < self.n_clusters:
                self.n_clusters = max(1, len(embeddings))

            kmeans_labels = KMeans(n_clusters=self.n_clusters, random_state=42).fit_predict(embeddings)

            cluster_map = {}
            for i, label in enumerate(kmeans_labels):
                if label not in cluster_map:
                    cluster_map[label] = []
                cluster_map[label].append(vc_names[i])

            clusters = []
            for cluster_id, members in cluster_map.items():
                cluster_summaries = "\n".join([f"{m}: {summaries_dict[m]}" for m in members])
                description = self.describe_cluster(cluster_summaries)

                clusters.append({
                    "cluster_id": cluster_id,
                    "description": description,
                    "members": members
                })

            return clusters

        except Exception as e:
            logging.error(f"Categorization failed: {e}")
            return []

    def describe_cluster(self, text_block):
        try:
            prompt = f"""
You are an investment intelligence analyst. Provide a 1-2 sentence summary that describes what these VC firms have in common based on the following data.

{text_block}

Return only the summary description.
"""
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Failed to describe cluster: {e}")
            return "No description available."
