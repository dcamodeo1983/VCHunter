from agents.utils import clean_text
from PyPDF2 import PdfReader
import logging

logging.basicConfig(level=logging.INFO)


class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, uploaded_file) -> str:
        try:
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            logging.info(f"✅ Extracted {len(text)} characters from uploaded file.")
            return clean_text(text)
        except Exception as e:
            logging.error(f"❌ Error reading file: {e}")
            return f"Error reading file: {e}"


class VCHunterOrchestrator:
    def __init__(self, agents):
        self.scraper = agents['scraper']
        self.portfolio_enricher = agents['portfolio']
        self.summarizer = agents['summarizer']
        self.embedder = agents['embedder']
        self.categorizer = agents['categorizer']
        self.relationship = agents['relationship']
        self.visualizer = agents['visualizer']
        self.matcher = agents['matcher']
        self.chatbot = agents['chatbot']
        self.gap = agents['gap']
        self.similar = agents['similar']

    def run(self, founder_text: str):
        logging.info("📝 Summarizing founder input...")
        founder_summary = self.summarizer.summarize_founder(founder_text)

        logging.info("📐 Generating founder embedding...")
        founder_embeds = self.embedder.embed([founder_summary])
        if not founder_embeds:
            raise ValueError("Founder embedding failed.")
        founder_vec = founder_embeds[0]

        vc_urls = [
            "https://a16z.com", "https://www.benchmark.com", "https://www.accel.com",
            "https://www.sequoiacap.com", "https://www.greylock.com", "https://www.indexventures.com",
            "https://www.baincapitalventures.com", "https://www.generalcatalyst.com", "https://www.ivp.com"
        ]

        vc_summaries = {}
        vc_portfolios = {}

        for url in vc_urls:
            try:
                logging.info(f"🌐 Scraping {url}")
                result = self.scraper.scrape(url)
                site_text = "\n".join(result["site_text"].values())
                enriched_texts = self.portfolio_enricher.enrich(result["portfolio_links"])
                portfolio_text = "\n".join(enriched_texts.values())

                logging.info(f"✏️ Summarizing site + portfolio text for {url}")
                summary = self.summarizer.summarize(site_text, portfolio_text)

                vc_summaries[url] = summary
                vc_portfolios[url] = list(enriched_texts.keys())
            except Exception as e:
                logging.warning(f"⚠️ Error processing {url}: {e}")
                continue

        logging.info("📐 Embedding VC summaries...")
        embeddings = self.embedder.embed([vc_summaries[url] for url in vc_summaries])
        vc_to_vectors = dict(zip(vc_summaries.keys(), embeddings))
        vc_to_companies = vc_portfolios

        logging.info("🧱 Categorizing VCs into clusters...")
        clusters = self.categorizer.categorize(embeddings, list(vc_summaries.keys()), vc_summaries)
        cluster_map = {vc: cluster['cluster_id'] for cluster in clusters for vc in cluster['members']}

        logging.info("🔗 Mapping relationships between VC firms...")
        relationship_graph = self.relationship(vc_to_companies, vc_to_vectors).analyze()

        logging.info("🧭 Visualizing VC map...")
        visuals = self.visualizer.plot_all(embeddings, list(vc_summaries.keys()), clusters, relationship_graph)

        logging.info("
