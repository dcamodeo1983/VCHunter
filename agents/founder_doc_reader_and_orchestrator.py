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
            return f"Error reading file: {e}"


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
        self.visualizer = agents['visualizer']

    def run(self, founder_text: str, trigger_nvca=False):
        # Step 1: Gather VC URLs
        if trigger_nvca:
            vc_urls = self.nvca_updater.scrape_nvca()
        else:
            vc_urls = self.scraper.scrape()

        # Step 2: Scrape VC sites
        raw_site_texts = self.scraper.extract_all(vc_urls)

        # Step 3: Enrich with portfolio data
        portfolio_data = self.portfolio_enricher.enrich(vc_urls)

        # Step 4: Summarize website + portfolio
        summaries = {}
        for url in vc_urls:
            site_text = raw_site_texts.get(url, "")
            portfolio_text = portfolio_data.get(url, "")
            summary = self.summarizer.summarize(site_text, portfolio_text)
            summaries[url] = summary

        # Step 5: Embed the VC summaries
        embedded_vectors = self.embedder.embed([summaries[url] for url in vc_urls])

        # Step 6: Categorize VC firms
        cluster_results = self.categorizer.categorize(
            embeddings=embedded_vectors,
            vc_names=vc_urls,
            summaries_dict=summaries
        )
        cluster_map = self.categorizer.cluster_map

        # Step 7: Relationship Graph
        graph = self.relationship.analyze(
            vc_to_companies=vc_urls,
            vc_to_vectors=dict(zip(vc_urls, embedded_vectors))
        )

        # Step 8: Visualization
        self.visualizer.render_graph(graph)

        # Step 9: Return outputs
        return {
            "summaries": summaries,
            "clusters": cluster_results,
            "graph": graph,
            "vc_urls": vc_urls,
            "embeddings": embedded_vectors,
            "cluster_map": cluster_map
        }
