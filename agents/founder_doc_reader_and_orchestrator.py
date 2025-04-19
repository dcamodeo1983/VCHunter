# 📄 FounderDocReaderAgent – Extract text from uploaded PDF or TXT
import numpy as np
from PyPDF2 import PdfReader
import streamlit as st

class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            return f"Error reading file: {e}"


# 🔁 VCHunterOrchestrator – Connects All Agents for Full Workflow
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

    def run(self, founder_text: str, trigger_nvca=False):
        vc_list = self.nvca_updater.fetch_vc_records() if trigger_nvca else []

        vc_summaries = []
        vc_names = []
        vc_embeddings = []
        vc_portfolios = {}
        vc_to_companies = {}

        for vc in vc_list:
            st.markdown(f"### 🔍 Scraping {vc['name']}")
            result = self.scraper.scrape(vc['url'])

            site_text = result.get("site_text", {})
            portfolio_links = result.get("portfolio_links", [])
            st.write("📄 Site text keys:", list(site_text.keys()))
            st.write("🔗 Portfolio links found:", len(portfolio_links))

            if not site_text:
                st.warning(f"⚠️ No site text scraped for {vc['name']} — skipping.")
                continue

            merged_site_text = " ".join(site_text.values())
            portfolio_data = self.portfolio_enricher.enrich(portfolio_links)
            merged_portfolio = "\n\n".join(portfolio_data.values())

            summary = self.summarizer.summarize(merged_site_text, merged_portfolio)
            st.text(f"🧠 VC Summary for {vc['name']}: {summary[:300] if summary else 'None'}")

            if summary:
                vc_summaries.append(summary)
                vc_names.append(vc['name'])
                vc_portfolios[vc['name']] = merged_portfolio
                vc_to_companies[vc['name']] = portfolio_links

        if not vc_summaries:
            return {
                "summaries": [],
                "clusters": [],
                "relationships": {},
                "matches": [],
                "gaps": []
            }

        texts = [str(s) for s in vc_summaries]
        vc_embeddings = self.embedder.embed(texts)

        if len(vc_embeddings) == 0:
            return {
                "summaries": vc_summaries,
                "clusters": [],
                "relationships": {},
                "matches": [],
                "gaps": []
            }

        clusters = self.categorizer.categorize(vc_embeddings, vc_names, {n: str(s) for n, s in zip(vc_names, vc_summaries)})

        cluster_vectors = {
            cid: np.mean([vc_embeddings[vc_names.index(v)] for v in cluster["members"]], axis=0)
            for cluster in clusters
            for cid in [cluster["cluster_id"]]
        }
        centroids = np.stack(list(cluster_vectors.values())) if cluster_vectors else np.array([])
        labels = list(cluster_vectors.keys())

        rel_agent = self.relationship(vc_to_companies, {name: vec for name, vec in zip(vc_names, vc_embeddings)})
        relationships = rel_agent.analyze()

        founder_embedding = self.embedder.embed([founder_text])[0]

        matches = self.matcher.match(founder_embedding, vc_embeddings, vc_names, {v: c['cluster_id'] for c in clusters for v in c['members']})
        gaps = self.gap.detect(founder_embedding, centroids, labels) if len(centroids) > 0 else []

        return {
            "summaries": vc_summaries,
            "clusters": clusters,
            "relationships": relationships,
            "matches": matches,
            "gaps": gaps
        }
