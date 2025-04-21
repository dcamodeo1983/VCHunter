import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


class RelationshipAgent:
    def __init__(self, vc_to_companies: Dict[str, List[str]], vc_to_vectors: Dict[str, np.ndarray]):
        """
        Args:
            vc_to_companies: Dictionary of {VC firm name: [list of portfolio company URLs/domains]}
            vc_to_vectors: Dictionary of {VC firm name: vector representation}
        """
        self.vc_to_companies = vc_to_companies
        self.vc_to_vectors = vc_to_vectors

    def analyze(self) -> Dict[str, List[Dict[str, Any]]]:
        relationships = []
        vcs = list(self.vc_to_companies.keys())

        for i in range(len(vcs)):
            for j in range(i + 1, len(vcs)):
                a, b = vcs[i], vcs[j]
                set_a = set(self.vc_to_companies.get(a, []))
                set_b = set(self.vc_to_companies.get(b, []))

                if not set_a and not set_b:
                    continue

                shared = set_a & set_b
                union = set_a | set_b
                jaccard = len(shared) / len(union) if union else 0.0

                # Cosine similarity between VC embeddings
                cosine_sim = 0.0
                try:
                    vec_a = self.vc_to_vectors[a]
                    vec_b = self.vc_to_vectors[b]
                    if vec_a is not None and vec_b is not None:
                        cosine_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
                except Exception as e:
                    logging.warning(f"Cosine similarity failed for {a} and {b}: {e}")

                relationships.append({
                    "firm_a": a,
                    "firm_b": b,
                    "shared_companies": list(shared),
                    "score": round(jaccard, 3),
                    "cosine_similarity": round(cosine_sim, 3),
                    "type": self._classify(jaccard)
                })

        return {"co_investment": relationships}

    def _classify(self, score: float) -> str:
        if score > 0.3:
            return "Strong Collaborators"
        elif score > 0.15:
            return "Occasional Co-Investors"
        elif score > 0.05:
            return "Loose Competitors"
        else:
            return "No Significant Overlap"
