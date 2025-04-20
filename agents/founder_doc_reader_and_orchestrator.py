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
        self.visualizer = agents['visualizer']

    def run(self, founder_text: str, trigger_nvca=False):
        # Hardcoded fallback VC URLs
        hardcoded_vcs = [
            {"name": "Sequoia Capital", "
