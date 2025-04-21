# agents/similar_company_agent.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarCompanyAgent:
    def __init__(self, embedder):
        """
        Args:
            embedder: An instance of EmbedderAgent used to embed company descriptions.
        """
        self.embedder = embedder

    def find_similar(self, founder_vec, portfolio_texts: dict, company_to_vcs: dict, top_n=5):
        """
        Identify companies in VC portfolios that are similar to the founder's business concept.

        Args:
            founder_vec: The embedded vector representation of the founder's summary.
            portfolio_texts: A dict of {company_url: combined website + portfolio text}.
            company_to_vcs: A dict of {company_url: [list of VC firms that invested]}.
            top_n: Number of top similar companies to return.

        Returns:
            A list of dicts containing company_url, similarity_score, and invested_vcs.
        """
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
