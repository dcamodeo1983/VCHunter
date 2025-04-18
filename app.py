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

# === Import Agent Classes ===
from agents.website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.relationship_agent import RelationshipAgent
from agents.categorizer_agent import CategorizerAgent
from agents.visualization_agent import VisualizationAgent
from agents.llm_embed_gap_match_chat import (
    LLMSummarizerAgent,
    EmbedderAgent,
    GapAnalysisAgent,
    FounderMatchAgent,
    ChatbotAgent
)
from agents.nvca_updater_agent import NVCAUpdaterAgent

# === Streamlit App UI ===
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter - Founder Intelligence Explorer")

uploaded_file = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["txt", "pdf"])
run_pipeline = st.button("Run VC Intelligence Analysis")
trigger_nvca = st.checkbox("Re-scrape NVCA Directory", value=False)

if uploaded_file and run_pipeline:
    with st.spinner("Running full analysis..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Step 1: Extract founder text
        reader = FounderDocReaderAgent()
        founder_text = reader.extract_text(file_path)

        # Step 2: Initialize agents
        agents = {
            "nvca": NVCAUpdaterAgent(),
            "scraper": VCWebsiteScraperAgent(),
            "portfolio": PortfolioEnricherAgent(),
            "summarizer": LLMSummarizerAgent(api_key=openai.api_key),
            "embedder": EmbedderAgent(api_key=openai.api_key),
            "categorizer": CategorizerAgent(api_key=openai.api_key),
            "relationship": RelationshipAgent,
            "matcher": FounderMatchAgent(),
            "gap": GapAnalysisAgent(),
            "chatbot": ChatbotAgent(api_key=openai.api_key)
        }

        # Step 3: Run full pipeline
        orchestrator = VCHunterOrchestrator(agents)
        results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca)

        st.success("✅ Analysis complete!")

        # === Display outputs ===
        st.header("🧠 VC Summaries")
        st.write(f"Processed {len(results['summaries'])} VC profiles.")
        for summary in results["summaries"]:
            st.json(summary)

        st.header("🎯 Top VC Matches")
        st.table(results["matches"])

        st.header("🌌 White Space / Gap Analysis")
        for gap in results["gaps"]:
            st.markdown(f"- **{gap['category']}** — Similarity: `{gap['score']:.3f}`")
            st.caption(gap["insight"])
