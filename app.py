import streamlit as st
import openai
import os
import tempfile

from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator

# === Load OpenAI API key ===
if "openai" in st.secrets:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    st.error("❌ OpenAI API key not found. Please add it in Streamlit Cloud → Settings → Secrets.")
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

        # ✅ FIXED LINE
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Step 1: Extract founder text
        reader = FounderDocReaderAgent()
        founder_text = reader.extract_text(file_path)

        # Step 2: Initialize agents
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

        # Step 3: Run the orchestration pipeline
        orchestrator = VCHunterOrchestrator(agents)

        # ✅ Optional live preview
        st.info("📡 Running agent pipeline...")
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
