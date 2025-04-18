# 🤝 RelationshipAgentV2 – Combines Portfolio Overlap and Embedding Similarity

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class RelationshipAgentV2:
    def __init__(self, vc_to_companies: Dict[str, List[str]], vc_embeddings: Dict[str, np.ndarray]):
        """
        Parameters:
        - vc_to_companies: {VC name: [portfolio company names/domains]}
        - vc_embeddings: {VC name: embedding vector (np.ndarray)}
        """
        self.vc_to_companies = vc_to_companies
        self.vc_embeddings = vc_embeddings
        self.relationships = {}

    def compute_overlap_score(self, a: str, b: str) -> Dict:
        set_a = set(self.vc_to_companies.get(a, []))
        set_b = set(self.vc_to_companies.get(b, []))
        shared = set_a & set_b
        total = set_a | set_b
        jaccard = len(shared) / len(total) if total else 0

        return {
            "shared": list(shared),
            "a_unique": list(set_a - shared),
            "b_unique": list(set_b - shared),
            "jaccard": jaccard,
            "count": len(shared)
        }

    def compute_embedding_similarity(self, a: str, b: str) -> float:
        try:
            vec_a = self.vc_embeddings[a].reshape(1, -1)
            vec_b = self.vc_embeddings[b].reshape(1, -1)
            return cosine_similarity(vec_a, vec_b)[0][0]
        except Exception as e:
            logging.warning(f"Similarity error between {a} and {b}: {e}")
            return 0.0

    def classify_relationship(self, jaccard: float, shared: int, diff: int) -> str:
        if shared == 0:
            return "No Relationship"
        if jaccard > 0.3:
            return "Strong Collaborators"
        elif jaccard > 0.15:
            return "Occasional Co-Investors"
        elif shared > 0 and diff < 3:
            return "Direct Competitors"
        else:
            return "Weak or Mixed Signal"

    def analyze(self) -> Dict[Tuple[str, str], Dict]:
        vcs = list(self.vc_to_companies.keys())
        matrix = {}

        for i in range(len(vcs)):
            for j in range(i + 1, len(vcs)):
                a, b = vcs[i], vcs[j]
                overlap = self.compute_overlap_score(a, b)
                emb_sim = self.compute_embedding_similarity(a, b)
                diff = abs(len(self.vc_to_companies.get(a, [])) - len(self.vc_to_companies.get(b, [])))

                relationship_type = self.classify_relationship(
                    overlap["jaccard"], overlap["count"], diff
                )

                matrix[(a, b)] = {
                    "type": relationship_type,
                    "embedding_score": round(emb_sim, 3),
                    "jaccard_score": round(overlap["jaccard"], 3),
                    "shared_companies": overlap["shared"][:5],
                    "more_than_5": len(overlap["shared"]) > 5
                }

        self.relationships = matrix
        return matrix

    def get_relationship_summary(self, vc1: str, vc2: str) -> str:
        rel = self.relationships.get((vc1, vc2)) or self.relationships.get((vc2, vc1))
        if not rel:
            return f"No observed relationship between {vc1} and {vc2}."
        more = "..." if rel['more_than_5'] else ""
        shared_str = ", ".join(rel['shared_companies']) + more
        return (
            f"{vc1} and {vc2} are classified as **{rel['type']}**.\n"
            f"They share {len(rel['shared_companies'])} portfolio companies.\n"
            f"Shared: {shared_str}\n"
            f"Jaccard: {rel['jaccard_score']} | Embedding Similarity: {rel['embedding_score']}"
        )

