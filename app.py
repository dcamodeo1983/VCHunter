import os
import streamlit as st
from dotenv import load_dotenv

# Import agents
from agents.founder_doc_reader_and_orchestrator import VCHunterOrchestrator, FounderDocReaderAgent
from agents.llm_embed_gap_match_chat import (
    ChatbotAgent, FounderMatchAgent, EmbedderAgent, LLMSummarizerAgent, GapAnalysisAgent
)
from agents.categorizer_agent import CategorizerAgent
from agents.relationship_agent import RelationshipAgent
from agents.visualization_agent import VisualizationAgent
from agents.similar_company_agent import SimilarCompanyAgent
from agents.website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI config
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter App")
st.markdown("Upload one or more startup concept documents and run a full VC landscape analysis.")

# Session state for documents
if "founder_docs" not in st.session_state:
    st.session_state.founder_docs = []

uploaded_file = st.file_uploader("📄 Upload a Startup Concept File (PDF, TXT, or DOCX)", type=["pdf", "txt", "docx"])
if uploaded_file:
    st.session_state.founder_docs.append(uploaded_file)
    st.success(f"Added: {uploaded_file.name}")

# Display uploaded docs
if st.session_state.founder_docs:
    st.markdown("### 📚 Uploaded Documents")
    for doc in st.session_state.founder_docs:
        st.markdown(f"- {doc.name}")

# Run analysis button
if st.session_state.founder_docs and st.button("🚀 Run Analysis"):
    st.info("Running full intelligence pipeline...")

    try:
        # Agents
        reader = FounderDocReaderAgent()
        summarizer = LLMSummarizerAgent(api_key=openai_api_key)
        embedder = EmbedderAgent(api_key=openai_api_key)
        categorizer = CategorizerAgent(api_key=openai_api_key)
        relationship = RelationshipAgent
        visualizer = VisualizationAgent()
        matcher = FounderMatchAgent()
        chatbot = ChatbotAgent(api_key=openai_api_key)
        gap = GapAnalysisAgent()
        scraper = VCWebsiteScraperAgent()
