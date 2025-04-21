import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import logging

class CategorizerAgent:
    def __init__(self, api_key, n_clusters=5):
        self.client = OpenAI(api_key=api_key)
        self.n_clusters = n_clusters
        self.cluster_map = {}

    def dynamic_cluster(self, embeddings):
        try:
            if len(embeddings) < 3:
                logging.info("Using KMeans due to small dataset")
                return KMeans(n_clusters=1, random_state=42).fit_predict(embeddings)

            # Try Agglomerative and KMeans, pick one with better silhouette
            k_labels = KMeans(n_clusters=min(self.n_clusters, len(embeddings)), random_state=42).fit_predict(embeddings)
            a_labels = AgglomerativeClustering(n_clusters=min(self.n_clusters, len(embeddings))).fit_predict(embeddings)

            k_score = silhouette_score(embeddings, k_labels)
            a_score = silhouette_score(embeddings, a_labels)

            logging.info(f"KMeans Silhouette: {k_score:.3f}, Agglomerative Silhouette: {a_score:.3f}")
            return k_labels if k_score >= a_score else a_labels

        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            return [0] * len(embeddings)

    def categorize(self, embeddings, vc_names, summaries_dict):
        try:
            labels = self.dynamic_cluster(embeddings)

            self.cluster_map = {}
            for i, label in enumerate(labels):
                if label not in self.cluster_map:
                    self.cluster_map[label] = []
                self.cluster_map[label].append(vc_names[i])

            clusters = []
            for cluster_id, members in self.cluster_map.items():
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
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Failed to describe cluster: {e}")
            return "No description available."
