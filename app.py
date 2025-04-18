import streamlit as st
import openai
import os
import tempfile

from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator

# === Load OpenAI API key ===
if "openai" in st.secrets:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# === Initialize agents ===
from agents.website_scraper_agent import VCWebsiteScraperAgentV2
from agents.portfolio_enricher_agent import PortfolioEnricherAgentV3
from agents.relationship_agent import RelationshipAgentV2
from agents.categorizer_agent import CategorizerAgentV2
from agents.visualization_agent import VisualizationAgentV2
from agents.llm_embed_gap_match_chat import (
    LLMSummarizerAgentV2,
    EmbedderAgentV2,
    GapAnalysisAgentV2,
    FounderMatchAgentV2,
    ChatbotAgentV2
)
from agents.nvca_updater_agent import NVCAUpdaterAgentV2

# === Streamlit App UI ===
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter - Founder Intelligence Explorer")

uploaded_file = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["txt", "pdf"])
run_pipeline = st.button("Run VC Intelligence Analysis")
trigger_nvca = st.checkbox("Re-scrape NVCA Directory", value=False)

if uploaded_file and run_pipeline:
    with st.spinner("Running full analysis..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False)
