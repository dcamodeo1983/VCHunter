import numpy as np
import logging
from openai import OpenAI
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan

class CategorizerAgent:
    def __init__(self, api_key, n_clusters=5):
        self.client = OpenAI(api_key=api_key)
        self.n_clusters = n_clusters
        self.cluster_map = {}

    def categorize(self, embeddings, vc_names, summaries_dict):
        try:
            labels = self.cluster_embeddings(embeddings)

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

    def cluster_embeddings(self, embeddings):
        N = len(embeddings)

        if N < 100:
            return self._run_kmeans(embeddings)
        elif 100 <= N < 500:
            try:
                labels, score = self._run_kmeans(embeddings, return_score=True)
                if score < 0.25:
                    raise ValueError("Low silhouette score")
                return labels
            except Exception as e:
                logging.warning(f"KMeans failed or low quality: {e}, falling back to HDBSCAN")
                return self._run_hdbscan(embeddings)
        else:
            return self._run_ensemble_clustering(embeddings)

    def _run_kmeans(self, embeddings, return_score=False):
        k = min(self.n_clusters, len(embeddings))
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(embeddings)
        if return_score:
            score = silhouette_score(embeddings, labels)
            return labels, score
        return labels

    def _run_hdbscan(self, embeddings):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, len(embeddings) // 20))
        return clusterer.fit_predict(embeddings)

    def _run_ensemble_clustering(self, embeddings):
        km_labels = self._run_kmeans(embeddings)
        ag_labels = AgglomerativeClustering(n_clusters=min(self.n_clusters, len(embeddings))).fit_predict(embeddings)
        hdb_labels = self._run_hdbscan(embeddings)

        final_labels = []
        for i in range(len(embeddings)):
            votes = [km_labels[i], ag_labels[i], hdb_labels[i]]
            final_labels.append(max(set(votes), key=votes.count))  # majority vote
        return np.array(final_labels)

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
