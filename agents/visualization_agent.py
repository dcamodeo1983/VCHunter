import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

class VisualizationAgent:
    def plot_all(self, embeddings, names, clusters, relationship_map):
        visuals = {}

        labels = [f"Cluster {c['cluster_id']}" for c in clusters for _ in c["members"]]
        flat_names = [name for c in clusters for name in c["members"]]
        le = LabelEncoder()
        color_labels = le.fit_transform(labels)

        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        fig1, ax1 = plt.subplots(figsize=(10, 7))
        scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], c=color_labels, s=100)
        for i, txt in enumerate(flat_names):
            ax1.annotate(txt[:10], (reduced[i, 0], reduced[i, 1]), fontsize=7)
        ax1.set_title("VC Cluster Map")
        visuals["cluster_map"] = fig1

        # Heatmap
        n = len(names)
        matrix = np.zeros((n, n))
        name_idx = {name: i for i, name in enumerate(names)}
        for rel_type, pairs in relationship_map.items():
            for p in pairs:
                i, j = name_idx.get(p["firm_a"]), name_idx.get(p["firm_b"])
                if i is not None and j is not None:
                    matrix[i][j] = p["score"]
                    matrix[j][i] = p["score"]
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=names, yticklabels=names, cmap="coolwarm", ax=ax2)
        ax2.set_title("VC Relationship Heatmap")
        visuals["heatmap"] = fig2

        # Network
        G = nx.Graph()
        for rel_type, pairs in relationship_map.items():
            for rel in pairs:
                G.add_edge(rel["firm_a"], rel["firm_b"], weight=rel["score"])
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=700, font_size=8, width=1.0, ax=ax3)
        ax3.set_title("VC Relationship Graph")
        visuals["network"] = fig3

        return visuals
