from collections import defaultdict
import logging
import numpy as np

class RelationshipAgent:
    def __init__(self, vc_to_companies: dict, vc_to_vectors: dict):
        """
        Args:
            vc_to_companies: Dictionary of {VC firm name: [list of portfolio company URLs/domains]}
            vc_to_vectors: Dictionary of {VC firm name: vector representation}
        """
        self.vc_to_companies = vc_to_companies
        self.vc_to_vectors = vc_to_vectors

    def analyze(self):
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

                # Optional: embed cosine similarity between VCs
                cosine_sim = 0.0
                try:
                    vec_a = self.vc_to_vectors[a]
                    vec_b = self.vc_to_vectors[b]
                    cosine_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
                except Exception as e:
                    logging.warning(f"Cosine similarity failed for {a} and {b}: {e}")

                relationship = {
                    "firm_a": a,
                    "firm_b": b,
                    "shared_companies": list(shared),
                    "score": round(jaccard, 3),
                    "cosine_similarity": round(cosine_sim, 3),
                    "type": self._classify(jaccard)
                }
                relationships.append(relationship)

        return {"co_investment": relationships}

    def _classify(self, score):
        if score > 0.3:
            return "Strong Collaborators"
        elif score > 0.15:
            return "Occasional Co-Investors"
        elif score > 0.05:
            return "Loose Competitors"
        else:
            return "No Significant Overlap"
