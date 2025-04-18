# 📊 VisualizationAgentV2 – Multi-view VC Landscape Mapper

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from typing import Dict, List

class VisualizationAgentV2:
    def __init__(
        self,
        embeddings: np.ndarray,
        vc_to_cluster: Dict[str, int],
        cluster_descriptions: Dict[int, str],
        vc_data: Dict[str, str],
        relationship_pairs: List[Dict[str, any]]
    ):
        self.embeddings = embeddings
        self.vc_to_cluster = vc_to_cluster
        self.cluster_descriptions = cluster_descriptions
        self.vc_data = vc_data
        self.relationship_pairs = relationship_pairs

    def plot_cluster_bubbles(self):
        """Visualize VC clusters using t-SNE"""
        vc_names = list(self.vc_to_cluster.keys())
        cluster_labels = [self.vc_to_cluster[vc] for vc in vc_names]
        le = LabelEncoder()
        color_labels = le.fit_transform(cluster_labels)

        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        reduced = tsne.fit_transform(self.embeddings)

        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "vc": vc_names,
            "cluster": [f"Cluster {c}" for c in cluster_labels]
        })

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df, x="x", y="y", hue="cluster", s=200, alpha=0.8)
        for i in range(len(df)):
            plt.text(df.x[i], df.y[i], df.vc[i][:10], fontsize=8)
        plt.title("VC Landscape (t-SNE Projection)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_relationship_heatmap(self):
        """Heatmap of pairwise VC relationships based on co-investment or similarity"""
        firms = list(self.vc_data.keys())
        n = len(firms)
        matrix = np.zeros((n, n))
        name_to_idx = {name: i for i, name in enumerate(firms)}

        for pair in self.relationship_pairs:
            a, b, score = pair["firm_a"], pair["firm_b"], pair["score"]
            if a in name_to_idx and b in name_to_idx:
                i, j = name_to_idx[a], name_to_idx[b]
                matrix[i][j] = score
                matrix[j][i] = score

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=firms, yticklabels=firms, cmap="coolwarm")
        plt.title("VC Relationship Heatmap")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_relationship_network(self):
        """Network graph showing VC firm collaborations or competition"""
        G = nx.Graph()
        for rel in self.relationship_pairs:
            a, b, score = rel["firm_a"], rel["firm_b"], rel["score"]
            G.add_edge(a, b, weight=score, type=rel.get("type", "unspecified"))

        pos = nx.spring_layout(G, k=0.5, seed=42)
        plt.figure(figsize=(12, 8))

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=9)

        plt.title("VC Relationship Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def run_all(self):
        self.plot_cluster_bubbles()
        self.plot_relationship_heatmap()
        self.plot_relationship_network()

