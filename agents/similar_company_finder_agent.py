import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarCompanyFinderAgent:
    def __init__(self, embedder):
        self.embedder = embedder  # Should be an instance of EmbedderAgent

    def find_similar(self, founder_vec, portfolio_texts: dict, company_to_vcs: dict, top_n=5):
        companies = list(portfolio_texts.keys())
        texts = [portfolio_texts[c] for c in companies]

        embeddings = self.embedder.embed(texts)
        if not embeddings or founder_vec is None:
            return []

        similarities = cosine_similarity([founder_vec], embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in ranked_indices:
            company = companies[idx]
            score = round(float(similarities[idx]), 3)
            vcs = company_to_vcs.get(company, [])
            results.append({
                "company_url": company,
                "similarity_score": score,
                "invested_vcs": vcs
            })

        return results
