import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class SimilarCompanyAgent:
    def __init__(self, embedder):
        """
        Args:
            embedder: An instance of EmbedderAgent with an .embed(texts) method.
        """
        self.embedder = embedder

    def find_similar(self, founder_vec, portfolio_texts: dict, company_to_vcs: dict, top_n=5):
        """
        Finds the most similar portfolio companies to the founder's startup idea.

        Args:
            founder_vec (np.ndarray): Embedding vector of the founder's idea.
            portfolio_texts (dict): Mapping from company URL to enriched text.
            company_to_vcs (dict): Mapping from company URL to list of VC firms that invested.
            top_n (int): Number of most similar companies to return.

        Returns:
            list of dicts containing 'company_url', 'similarity_score', and 'invested_vcs'
        """
        companies = list(portfolio_texts.keys())
        texts = [portfolio_texts[c] for c in companies]

        embeddings = self.embedder.embed(texts)
        if not embeddings or founder_vec is None:
            logging.warning("❌ No embeddings returned in SimilarCompanyAgent.")
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
