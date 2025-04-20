import streamlit as st
st.set_page_config(page_title="VC Hunter", layout="wide")  # ✅ Must be first Streamlit command

import tempfile
import os
import openai

from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator
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

# === Load OpenAI API key ===
if "openai" in st.secrets:
    openai.api_key = st.secrets["openai"]
else:
    st.warning("⚠️ No OpenAI API key found in Streamlit secrets.")
    st.stop()

# === App UI ===
st.title("🚀 VC Hunter - Founder Intelligence Explorer")
uploaded_file = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["pdf", "txt"])
trigger_nvca = st.checkbox("🔄 Refresh VC List from Web (slower)")

# === Instantiate agents ===
vc_list_agent = VCListAggregatorAgent()
scraper_agent = VCWebsiteScraperAgent()
portfolio_agent = PortfolioEnricherAgent()
summarizer_agent = LLMSummarizerAgent(api_key=openai.api_key)
embedder_agent = EmbedderAgent(api_key=openai.api_key)
gap_agent = GapAnalysisAgent()
matcher_agent = FounderMatchAgent()
chatbot_agent = ChatbotAgent(api_key=openai.api_key)
categorizer_agent = CategorizerAgent(api_key=openai.api_key)
#relationship_agent = RelationshipAgent()
visualizer_agent = VisualizationAgent()
reader_agent = FounderDocReaderAgent()

# === File handling + analysis ===
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
        'embedder': embedder_agent,
        'gap': gap_agent,
        'matcher': matcher_agent,
        'chatbot': chatbot_agent,
        'categorizer': categorizer_agent,
        'relationship': RelationshipAgent,
        'visualizer': visualizer_agent,
    }

    orchestrator = VCHunterOrchestrator(agents)

    st.info("🔍 Running full VC analysis…")
    results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca)
    st.success("✅ Analysis complete!")

    # === Display outputs ===
    st.subheader("🧠 Founder Summary")
    st.markdown(results["founder_summary"])

    if results["clusters"]:
        st.subheader("🔷 VC Clusters")
        for cluster in results["clusters"]:
            st.markdown(f"**Cluster {cluster['cluster_id']}** — {cluster['description']}")
            st.markdown(", ".join(cluster["members"]))

    if results["matches"]:
        st.subheader("🎯 Best Matched VCs")
        for match in results["matches"]:
            st.markdown(f"- **{match['vc']}** (score: {match['score']:.2f}) — Cluster {match['cluster']}")

    if results["gaps"]:
        st.subheader("🌌 Gap Opportunities")
        for gap in results["gaps"]:
            st.markdown(f"- **Cluster {gap['label']}** — {gap['reason']}")

    if results["visuals"]:
        st.subheader("📊 Visualizations")
        for key, fig in results["visuals"].items():
            st.pyplot(fig)

    if results["chatbot"]:
        st.subheader("💬 Ask the VC Analyst")
        user_question = st.text_input("Ask a question about the VC landscape:")
        if user_question:
            reply = results["chatbot"].ask(user_question)
            st.markdown(reply)
