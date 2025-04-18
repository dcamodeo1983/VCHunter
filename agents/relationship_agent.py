from collections import defaultdict

class RelationshipAgentV2:
    def __init__(self, vc_to_companies: dict, vc_to_vectors: dict):
        self.vc_to_companies = vc_to_companies
        self.vc_to_vectors = vc_to_vectors

    def analyze(self):
        relationships = []
        vcs = list(self.vc_to_companies.keys())

        for i in range(len(vcs)):
            for j in range(i + 1, len(vcs)):
                a, b = vcs[i], vcs[j]
                set_a = set(self.vc_to_companies[a])
                set_b = set(self.vc_to_companies[b])

                if not set_a and not set_b:
                    continue

                shared = set_a & set_b
                union = set_a | set_b
                jaccard = len(shared) / len(union) if len(union) > 0 else 0

                relationship = {
                    "firm_a": a,
                    "firm_b": b,
                    "shared_companies": list(shared),
                    "score": round(jaccard, 3),
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
