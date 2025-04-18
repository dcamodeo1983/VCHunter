import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

class VisualizationAgent:
    def __init__(self, embeddings, vc_to_cluster, cluster_descriptions, vc_data, relationship_map):
        self.embeddings = embeddings
        self.vc_to_cluster = vc_to_cluster
        self.cluster_descriptions = cluster_descriptions
        self.vc_data = vc_data
        self.relationship_map = relationship_map

    def plot_cluster_bubbles(self):
        if len(self.embeddings) < 2:
            print("🛑 Not enough data to plot clusters.")
            return

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
            plt.text(df.x[i], df.y[i], df.vc[i][:12], fontsize=8)
        plt.title("VC Landscape (t-SNE Projection)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_relationship_heatmap(self):
        firms = list(self.vc_data.keys())
        if not firms:
            print("🛑 No firm data available for heatmap.")
            return

        n = len(firms)
        matrix = np.zeros((
