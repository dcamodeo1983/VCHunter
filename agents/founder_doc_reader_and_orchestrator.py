# 📄 FounderDocReaderAgent – Extract text from uploaded PDF or TXT
import os
import logging
from PyPDF2 import PdfReader
from docx import Document

class FounderDocReaderAgent:
    def __init__(self):
        pass

    def extract_text(self, file_path: str) -> str:
        try:
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                raise ValueError("Unsupported file type.")
            return text.strip()
        except Exception as e:
            logging.error(f"Error reading founder file: {e}")
            return ""

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
        self.visualizer = agents['visualizer']

    def run(self, founder_text: str, trigger_nvca: bool = False) -> dict:
        try:
            # Step 1: Update VC list (optional refresh from public sources)
            if trigger_nvca:
                logging.info("Refreshing VC URLs from upstream...")
                vc_urls = self.nvca_updater.update()
            else:
                logging.info("Using existing scraped VC URLs...")
                vc_urls = self.scraper.get_urls()

            # Step 2: Enrich with portfolio links
            vc_to_companies = self.portfolio_enricher.enrich(vc_urls)

            # Step 3: Summarize VC firm sites
            vc_summaries = self.summarizer.summarize(vc_urls)

            # Step 4: Embed VC summaries
            vc_to_vectors = self.embedder.embed(vc_summaries)

            # Step 5: Categorize VCs with dynamic clustering
            cluster_descriptions = self.categorizer.categorize(
                embeddings=list(vc_to_vectors.values()),
                vc_names=list(vc_to_vectors.keys()),
                summaries_dict=vc_summaries
            )
            cluster_map = self.categorizer.cluster_map

            # Step 6: Analyze relationships
            relationship_graph = self.relationship.analyze(vc_to_companies, vc_to_vectors)

            # Step 7: Match founder to VCs
            founder_vec = self.embedder.embed([founder_text])[0]
            match_results = self.matcher.match(
                founder_vector=founder_vec,
                vc_vectors=list(vc_to_vectors.values()),
                vc_names=list(vc_to_vectors.keys()),
                cluster_map=cluster_map
            )

            # Step 8: Visualize relationship graph
            self.visualizer.render_graph(relationship_graph)

            # Step 9: Generate chatbot context
            chatbot_reply = self.chatbot.create(vc_summaries, founder_text)

            return {
                "summary": founder_text,
                "vc_summaries": vc_summaries,
                "match_results": match_results,
                "cluster_descriptions": cluster_descriptions,
                "chatbot_reply": chatbot_reply
            }

        except Exception as e:
            logging.error(f"VCHunter orchestrator failed: {e}")
            return {
                "summary": founder_text,
                "vc_summaries": {},
                "match_results": [],
                "cluster_descriptions": [],
                "chatbot_reply": "Error generating response."
            }
