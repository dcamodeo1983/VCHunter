import os
import streamlit as st
import openai

from agents.vc_list_aggregator_agent import VCListAggregatorAgent
from agents.website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.llm_embed_gap_match_chat import (
    LLMSummarizerAgent,
    EmbedderAgent,
    GapAnalysisAgent,
    FounderMatchAgent,
    ChatbotAgent
)
from agents.categorizer_agent import CategorizerAgent
from agents.relationship_agent import RelationshipAgent
from agents.visualization_agent import VisualizationAgent
from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter - Founder Intelligence Explorer")

# ✅ Load OpenAI API key from Streamlit secrets
if "openai_api_key" not in st.secrets or not st.secrets["openai_api_key"]:
    st.error("❌ Missing OpenAI API key. Please set it in Streamlit → Settings → Secrets → openai_api_key.")
    st.stop()

api_key = st.secrets["openai_api_key"]
openai.api_key = api_key
st.success("✅ OpenAI key loaded successfully.")

# === File upload ===
uploaded_file = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["txt", "pdf"])
csv_file = st.file_uploader("Upload VC URLs (CSV with 'url' column)", type="csv")

# === Optional refresh flag ===
trigger_nvca = st.checkbox("🔄 Refresh VC list from GitHub and Dealroom sources")

# === Initialize agents ===
nvca_agent = VCListAggregatorAgent()
scraper_agent = VCWebsiteScraperAgent()
portfolio_agent = PortfolioEnricherAgent()
summarizer_agent = LLMSummarizerAgent(api_key=api_key)
embedder_agent = EmbedderAgent(api_key=api_key)
gap_agent = GapAnalysisAgent()
matcher_agent = FounderMatchAgent()
chatbot_agent = ChatbotAgent(api_key=api_key)
categorizer_agent = CategorizerAgent(api_key=api_key)
visualizer_agent = VisualizationAgent()
reader_agent = FounderDocReaderAgent()

agents = {
    "nvca": nvca_agent,
    "scraper": scraper_agent,
    "portfolio": portfolio_agent,
    "summarizer": summarizer_agent,
    "embedder": embedder_agent,
    "gap": gap_agent,
    "matcher": matcher_agent,
    "chatbot": chatbot_agent,
    "categorizer": categorizer_agent,
    "visualizer": visualizer_agent
}

# === Run Pipeline ===
if uploaded_file:
    with st.spinner("⏳ Running analysis..."):
        try:
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            founder_text = reader_agent.extract_text(temp_path)

            if not founder_text:
                st.error("❌ No readable content extracted from the uploaded file.")
                st.stop()

            orchestrator = VCHunterOrchestrator(agents)
            results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca, csv=csv_file)

            st.success("✅ Analysis complete!")

            st.subheader("🧠 Founder Summary")
            st.markdown(results["founder_summary"] or "⚠️ No summary generated.")

            st.subheader("📊 VC Clusters")
            st.json(results["clusters"])

            st.subheader("🤝 VC Relationships")
            st.json(results["relationships"])

            st.subheader("🎯 Top Matches")
            st.json(results["matches"])

            st.subheader("📉 Gap Analysis")
            st.json(results["gaps"])

            st.subheader("🧬 Visual Landscape")
            st.pyplot(results["visuals"])

            st.subheader("💬 Ask the Chatbot")
            st.write("Coming soon: Interactive chat with founder & VC memory.")

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
