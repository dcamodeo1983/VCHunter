import streamlit as st
import os
import tempfile

from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator

# === OpenAI API Key from Streamlit secrets ===
import openai
openai.api_key = st.secrets["openai"]["api_key"]

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
    with st.spinner("Processing..."):

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Extract text from file
        reader = FounderDocReaderAgent()
        founder_text = reader.extract_text(file_path)

        # Initialize all agents
        agents = {
            "nvca": NVCAUpdaterAgentV2(),
            "scraper": VCWebsiteScraperAgentV2(),
            "portfolio": PortfolioEnricherAgentV3(),
            "summarizer": LLMSummarizerAgentV2(api_key=openai.api_key),
            "embedder": EmbedderAgentV2(api_key=openai.api_key),
            "categorizer": CategorizerAgentV2(api_key=openai.api_key),
            "relationship": RelationshipAgentV2,
            "matcher": FounderMatchAgentV2(),
            "gap": GapAnalysisAgentV2(),
            "chatbot": ChatbotAgentV2(api_key=openai.api_key)
        }

        orchestrator = VCHunterOrchestrator(agents)
        results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca)

        # === Display Results ===
        st.success("Analysis complete!")

        st.header("🧠 Summaries")
        for summary in results["summaries"]:
            st.json(summary)

        st.header("🎯 Top VC Matches")
        st.table(results["matches"])

        st.header("🌌 Opportunity Gaps")
        for gap in results["gaps"]:
            st.markdown(f"- **{gap['category']}** — Similarity: `{gap['score']:.3f}`")
            st.caption(gap["insight"])
