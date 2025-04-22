import logging
from agents.utils import clean_text
from PyPDF2 import PdfReader

class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, uploaded_file) -> str:
        try:
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            return clean_text(text)
        except Exception as e:
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
        logging.info("🔍 Summarizing founder text...")
        founder_summary = self.summarizer.summarize_founder(founder_text)
        logging.info("🧠 Generating founder embedding...")
        founder_embeds = self.embedder.embed([founder_summary])
        if not founder_embeds:
            raise ValueError("Founder embedding failed.")
        founder_vec = founder_embeds[0]

        vc_urls = [
    "https://a16z.com",
    "https://www.sequoiacap.com",
    "https://www.benchmark.com",
    "https://www.luxcapital.com",
    "https://www.8vc.com",
    "https://www.coatue.com",
    "https://www.indexventures.com",
    "https://www.felicis.com",
    "https://www.greylock.com",
    "https://www.accel.com",
    "https://www.lightspeedvp.com",
    "https://www.generalcatalyst.com",
    "https://www.ventureast.net",
    "https://www.dcvc.com",
    "https://www.crv.com",
    "https://www.ivp.com",
    "https://www.bvp.com",
    "https://www.union.vc",
    "https://www.cofoundpartners.com",
    "https://www.signal.vc"
]

        vc_summaries, vc_portfolios = {}, {}
        for url in vc_urls:
            try:
                logging.info(f"🌐 Scraping VC site: {url}")
                result = self.scraper.scrape(url)
                site_text = "\n".join(result["site_text"].values())
                logging.info(f"📚 Enriching portfolio for: {url}")
                enriched_texts = self.portfolio_enricher.enrich(result["portfolio_links"])
                portfolio_text = "\n".join(enriched_texts.values())
                logging.info(f"📝 Summarizing site and portfolio for: {url}")
                summary = self.summarizer.summarize(site_text, portfolio_text)

                if summary:
                    vc_summaries[url] = summary
                    vc_portfolios[url] = list(enriched_texts.keys())
            except Exception as e:
                logging.error(f"⚠️ Error processing {url}: {e}")
                continue

        if not vc_summaries:
            raise ValueError("No VC summaries could be processed.")

        logging.info("🔢 Embedding VC summaries...")
        embeddings = self.embedder.embed([vc_summaries[url] for url in vc_summaries])
        if not embeddings:
            raise ValueError("VC embeddings failed.")

        vc_to_vectors = dict(zip(vc_summaries.keys(), embeddings))
        vc_to_companies = vc_portfolios

        logging.info("📈 Categorizing VC firms...")
        clusters = self.categorizer.categorize(embeddings, list(vc_summaries.keys()), vc_summaries)
        cluster_map = {vc: cluster['cluster_id'] for cluster in clusters for vc in cluster['members']}

        logging.info("🔗 Analyzing co-investment and relationships...")
        relationship_graph = self.relationship(vc_to_companies, vc_to_vectors).analyze()

        logging.info("📊 Generating visualizations...")
        visuals = self.visualizer.plot_all(embeddings, list(vc_summaries.keys()), clusters, relationship_graph)

        logging.info("🎯 Matching VCs to founder...")
        matches = self.matcher.match(founder_vec, embeddings, list(vc_summaries.keys()), cluster_map)

        logging.info("🚪 Detecting whitespace/gap opportunities...")
        gap = self.gap.detect(founder_vec, embeddings, [c['cluster_id'] for c in clusters])

        logging.info("🔍 Finding similar portfolio companies...")
        similar = self.similar.find_similar(founder_vec, vc_to_vectors, vc_to_companies)

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
