from openai import OpenAI
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class CategorizerAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def categorize(self, vectors, names, descriptions, use_hdbscan=False):
        try:
            X = np.stack(vectors)

            if use_hdbscan:
                clusterer = HDBSCAN(min_cluster_size=3)
                labels = clusterer.fit_predict(X)
            else:
                clusterer = KMeans(n_clusters=5, random_state=42)
                labels = clusterer.fit_predict(X)

            output = []
            for cluster_id in np.unique(labels):
                cluster_names = [n for i, n in enumerate(names) if labels[i] == cluster_id]
                cluster_texts = [descriptions[n] for n in cluster_names]

                prompt = f"Summarize the investment style of this VC cluster:\n\n" + "\n\n".join(cluster_texts)

                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                summary = response.choices[0].message.content.strip()

                output.append({
                    "cluster_id": f"Cluster {cluster_id}",
                    "summary": summary,
                    "members": cluster_names
                })

            return output

        except Exception as e:
            logging.error(f"Categorization failed: {e}")
            return []
