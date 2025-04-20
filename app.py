import streamlit as st
import tempfile
import os
import openai

from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator
from agents.vc_list_aggregator_agent import VCListAggregatorAgent
from agents.website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent

# TEMP DEBUGGING BLOCK
import agents.llm_embed_gap_match_chat as test_mod
st.write("🔍 Contents of llm_embed_gap_match_chat module:")
st.write(dir(test_mod))

from agents.llm_embed_gap_match_chat import (
    LLMSummarizerAgent,
    # Other agents will be added back after test
)

# === Load OpenAI API key ===
if "openai" in st.secrets:
    openai.api_key = st.secrets["openai"]
else:
    st.warning("⚠️ No OpenAI API key found in Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter - Founder Intelligence Explorer")

uploaded_file = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["pdf", "txt"])
trigger_nvca = st.checkbox("🔄 Refresh VC List from Web (slower)")

vc_list_agent = VCListAggregatorAgent()
scraper_agent = VCWebsiteScraperAgent()
portfolio_agent = PortfolioEnricherAgent()
summarizer_agent = LLMSummarizerAgent(api_key=openai.api_key)

reader_agent = FounderDocReaderAgent()

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    founder_text = reader_agent.extract_text(file_path)

    agents = {
        'nvca': vc_list_agent,
        'scraper': scraper_agent,
        'portfolio': portfolio_agent,
        'summarizer': summarizer_agent,
        # Others omitted temporarily for testing
    }

    orchestrator = VCHunterOrchestrator(agents)
    st.info("🔍 Running full VC analysis…")
    results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca)
    st.success("✅ Complete")
