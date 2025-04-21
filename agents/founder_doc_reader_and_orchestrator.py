from founder_doc_reader_and_orchestrator import clean_text

class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, file_text: str) -> str:
        return clean_text(file_text)


class VCHunterOrchestrator:
    def __init__(self, agents: dict):
        self.scraper = agents["scraper"]
        self.portfolio = agents["portfolio"]
        self.summarizer = agents["summarizer"]
        self.embedder = agents["embedder"]
        self.categorizer = agents["categorizer"]
        self.relationship = agents["relationship"]
        self.matcher = agents["matcher"]
        self.gap = agents["gap"]
        self.chatbot = agents["chatbot"]
        self.visualizer = agents["visualizer"]
        self.similar = agents["similar"]

    def run(self, founder_summary, urls=None):
        if not urls:
            urls = [
                "https://a16z.com", "https://sequoiacap.com", "https://greylock.com",
                "https://lightspeedvp.com", "https://benchmark.com", "https://foundersfund.com"
            ]

        summaries, vectors, vc_to_companies = {}, [], {}
        names = []

        for url in urls:
            name = url.replace("https://", "").replace("www.", "").split(".")[0].title()
            scraped = self.scraper.scrape(url)
            enriched = self.portfolio.enrich(scraped["portfolio_links"])
            enriched_text = "\n".join(enriched.values())
            summary = self.summarizer.summarize(scraped["site_text"], enriched_text)
            summaries[name] = summary
            names.append(name)
            vc_to_companies[name] = list(enriched.keys())

        vectors = self.embedder.embed(list(summaries.values()))
        clusters = self.categorizer.categorize(vectors, names, summaries)
        cluster_map = {
            name: f"Cluster {cid}" for cid, cluster in enumerate(clusters) for name in cluster["members"]
        }

        rel_map = self.relationship(vc_to_companies, dict(zip(names, vectors))).analyze()
        visuals = self.visualizer.plot_all(vectors, names, clusters, rel_map)
        matches = self.matcher.match(founder_summary, vectors, names, cluster_map)
        gaps = self.gap.detect(founder_summary, vectors, names)
        similar = self.similar.find(founder_summary, vc_to_companies, summaries)

        return {
            "matches": matches,
            "gaps": gaps,
            "similar": similar,
            "visuals": visuals
        }

    def chat(self, question, founder_summary):
        return self.chatbot.create([], founder_summary)
