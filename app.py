import os
import streamlit as st
from dotenv import load_dotenv

from founder_doc_reader_and_orchestrator import VCHunterOrchestrator, FounderDocReaderAgent
from llm_embed_gap_match_chat import ChatbotAgent, FounderMatchAgent, EmbedderAgent, LLMSummarizerAgent, GapAnalysisAgent
from categorizer_agent import CategorizerAgent
from relationship_agent import RelationshipAgent
from visualization_agent import VisualizationAgent
from portfolio_enricher_agent import PortfolioEnricherAgent
from website_scraper_agent import VCWebsiteScraperAgent
from similar_company_finder_agent import SimilarCompanyFinderAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter App")
st.markdown("Upload your white paper and analyze venture capital relationships.")

uploaded_file = st.file_uploader("Upload Founder Document", type=["pdf", "txt", "docx", "pdf"])

if uploaded_file and openai_api_key:
    st.success("File uploaded. Running full VC analysis pipeline...")

    # Step 1: Read and summarize
    reader = FounderDocReaderAgent()
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    founder_text = reader.extract_text(raw_text)
    summarizer = LLMSummarizerAgent(api_key=openai_api_key)
    founder_summary = summarizer.summarize_founder(founder_text)

    st.subheader("📝 Founder Summary")
    st.write(founder_summary)

    # Step 2: Load and run orchestrator
    orchestrator = VCHunterOrchestrator({
        "nvca": None,  # Optional
        "scraper": VCWebsiteScraperAgent(),
        "portfolio": PortfolioEnricherAgent(limit=8),
        "summarizer": summarizer,
        "embedder": EmbedderAgent(api_key=openai_api_key),
        "categorizer": CategorizerAgent(api_key=openai_api_key),
        "relationship": RelationshipAgent,
        "matcher": FounderMatchAgent(),
        "gap": GapAnalysisAgent(),
        "chatbot": ChatbotAgent(api_key=openai_api_key),
        "visualizer": VisualizationAgent(),
        "similar": SimilarCompanyFinderAgent()
    })

    result = orchestrator.run(founder_summary)

    st.subheader("📊 Visual Insights")
    for label, fig in result["visuals"].items():
        st.pyplot(fig)

    st.subheader("💡 Top VC Matches")
    st.dataframe(result["matches"])

    st.subheader("🧭 Market Gaps (White Space)")
    st.json(result["gaps"])

    st.subheader("🔍 Similar Companies to Your Idea")
    for rec in result["similar"]:
        st.markdown(f"**{rec['company']}**")
        st.markdown(f"- Website: {rec['url']}")
        st.markdown(f"- Investors: {', '.join(rec['vcs'])}")
        st.markdown("---")

    st.subheader("💬 Chat")
    user_q = st.text_input("Ask a question about your idea or the VC landscape:")
    if user_q:
        reply = orchestrator.chat(user_q, founder_summary)
        st.write(reply)
