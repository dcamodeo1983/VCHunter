import numpy as np
from PyPDF2 import PdfReader
import logging

class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            logging.error(f"Error reading uploaded file: {e}")
            return ""


class VCHunterOrchestrator:
    def __init__(self, agents):
        self.nvca_updater = agents['nvca']
        self.scraper = agents['scraper']
        self.portfolio_enricher = agents['portfolio']
        self.summarizer = agents['summarizer']
        self.embedder = agents['embedder']
        self.categorizer = agents['categorizer']
        self.relationship = agents['relationship']
        self.matcher = agents['matcher']
        self.gap = agents['gap']
        self.chatbot = agents['chatbot']

    def _empty_result(self, vc_summaries=None, clusters=None, relationships=None):
        return {
            "summaries": vc_summaries or [],
            "clusters": clusters or [],
            "relationships": relationships or {},
            "matches": [],
            "gaps": []
        }

    def run(self, founder_text: str, trigger_nvca=False):
        vc_list = self.nvca_updater.fetch_vc_records() if trigger_nvca else []

        vc_summaries = []
        vc_names = []
        vc_embeddings = []
        vc_portfolios = {}
        vc_to_companies = {}

        for vc in vc_list:
            result = self.scraper.scrape(vc['url'])
            portfolio_data = self.portfolio_enricher.enrich(result['portfolio_links'])
            merged_portfolio = "\n\n".join(portfolio_data.values())

            summary = self.summarizer.summarize(" ".join(result['site_text'].values()), merged_portfolio)
            vc_summaries.append(summary)
            vc_names.append(vc['name'])
            vc_portfolios[vc['name']] = merged_portfolio
            vc_to_companies[vc['name']] = result['portfolio_links']

        if not vc_summaries:
            return self._empty_result()

        texts = [str(s) for s in vc_summaries]
        vc_embeddings = self.embedder.embed(texts)

        if not vc_embeddings:
            return self._empty_result(vc_summaries=vc_summaries)

        clusters = self.categorizer.categorize(
            vc_embeddings,
            vc_names,
            {n: str(s) for n, s in zip(vc_names, vc_summaries)}
        )

        cluster_vectors = {
            cid: np.mean([vc_embeddings[vc_names.index(v)] for v in cluster["members"]], axis=0)
            for cluster in clusters
            for cid in [cluster["cluster_id"]]
        }
        centroids = np.stack(list(cluster_vectors.values())) if cluster_vectors else np.array([])
        labels = list(cluster_vectors.keys())

        rel_agent = self.relationship(vc_to_companies, {name: vec for name, vec in zip(vc_names, vc_embeddings)})
        relationships = rel_agent.analyze()

        founder_vectors = self.embedder.embed([founder_text])
        if not founder_vectors:
            logging.warning("❌ Founder embedding failed — skipping matches and gaps.")
            return self._empty_result(vc_summaries=vc_summaries, clusters=clusters, relationships=relationships)

        founder_embedding = founder_vectors[0]

        matches = self.matcher.match(
            founder_embedding,
            vc_embeddings,
            vc_names,
            {v: c['cluster_id'] for c in clusters for v in c['members']}
        )

        gaps = self.gap.detect(founder_embedding, centroids, labels) if len(centroids) > 0 else []

        return {
            "summaries": vc_summaries,
            "clusters": clusters,
            "relationships": relationships,
            "matches": matches,
            "gaps": gaps
        }
