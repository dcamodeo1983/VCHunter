import logging
from agents.utils import clean_text
from PyPDF2 import PdfReader

# Setup module-level logger
logger = logging.getLogger(__name__)


class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, uploaded_file) -> str:
        try:
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            logger.info(f"Extracted {len(text)} characters from founder doc.")
            return clean_text(text)
        except Exception as e:
            logger.error(f"Error reading founder doc: {e}")
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
        logger.info("🚀 Starting orchestrator pipeline")

        founder_summary = self.summarizer.summarize_founder(founder_text)
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
                logger.info(f"🌐 Scraping and enriching: {url}")
                result = self.scraper.scrape(url)
                site_text = "\n".join(result["site_text"].values())
                enriched_texts = self.portfolio_enricher.enrich(result["portfolio_links"])
                portfolio_text = "\n".join(enriched_texts.values())
                summary = self.summarizer.summarize(site_text, portfolio_text)

                vc_summaries[url] = summary
                vc_portfolios[url] = list(enriched_texts.keys())
            except Exception as e:
                logger.warning(f"⚠️ Error processing {url}: {e}")
                continue

        embeddings = self.embedder.embed([vc_summaries[url] for url in vc_summaries])
        vc_to_vectors = dict(zip(vc_summaries.keys(), embeddings))
        vc_to_companies = vc_portfolios

        logger.info("📊 Categorizing VC firms")
        clusters = self.categorizer.categorize(embeddings, list(vc_summaries.keys()), vc_summaries)
        cluster_map = {vc: cluster['cluster_id'] for cluster in clusters for vc in cluster['members']}

        logger.info("🔗 Mapping relationships")
        relationship_graph = self.relationship(vc_to_companies, vc_to_vectors).analyze()

        logger.info("🖼 Generating visuals")
        visuals = self.visualizer.plot_all(embeddings, list(vc_summaries.keys()), clusters, relationship_graph)

        logger.info("🧠 Matching founder to VCs")
        matches = self.matcher.match(founder_vec, embeddings, list(vc_summaries.keys()), cluster_map)

        logger.info("🚪 Detecting strategic whitespace")
        gap = self.gap.detect(founder_vec, embeddings, [c['cluster_id'] for c in clusters])

        logger.info("🔍 Finding similar companies")
        similar = self.similar.find_similar(founder_vec, vc_to_vectors, vc_to_companies)

        logger.info("✅ Orchestration complete")
        return {
            "founder_summary": founder_summary,
            "vc_summaries": list(vc_summaries.values()),
            "clusters": clusters,
            "relationships": relationship_graph,
            "visuals": visuals,
            "matches": matches,
            "gap": gap,
            "similar_companies": similar
        }
