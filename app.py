import os
import logging
import streamlit as st
from dotenv import load_dotenv

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter App")
st.markdown("Upload one or more white papers to analyze startup fit, VC categories, co-investment networks, and portfolio signals.")

# Session state setup
if "founder_docs" not in st.session_state:
    st.session_state["founder_docs"] = []
if "results" not in st.session_state:
    st.session_state["results"] = None

# File uploader
uploaded_files = st.file_uploader("📄 Upload Startup Concept Documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)
if uploaded_files:
    st.session_state["founder_docs"] = uploaded_files
    st.success(f"{len(uploaded_files)} document(s) uploaded.")

# Run Analysis
if st.button("🚀 Run Analysis") and st.session_state["founder_docs"] and openai_api_key:
    try:
        st.info("⏳ Running full intelligence pipeline... This may take 1–3 minutes.")
        logger.info("🚀 Starting VC Hunter Analysis")

        # Initialize agents
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
        portfolio = PortfolioEnricherAgent()
        similar = SimilarCompanyAgent(embedder=embedder)

        agents = {
            "scraper": scraper,
            "portfolio": portfolio,
            "summarizer": summarizer,
            "embedder": embedder,
            "categorizer": categorizer,
            "relationship": relationship,
            "visualizer": visualizer,
            "matcher": matcher,
            "chatbot": chatbot,
            "gap": gap,
            "similar": similar
        }

        orchestrator = VCHunterOrchestrator(agents)

        # Combine all uploaded docs into one text blob
        full_text = ""
        for file in st.session_state["founder_docs"]:
            extracted = reader.extract_text(file)
            logger.info(f"📄 Extracted {len(extracted)} characters from uploaded file.")
            full_text += extracted + "\n"

        # Run orchestration
        results = orchestrator.run(full_text)
        st.session_state["results"] = results
        st.success("✔️ Analysis complete.")
        logger.info("✅ VC Hunter analysis completed successfully.")

    except Exception as e:
        logger.exception(f"❌ Pipeline execution failed: {e}")
        st.error(f"Pipeline execution failed: {e}")
        st.stop()

# Show results
results = st.session_state.get("results")
if results:
    st.subheader("📝 Founder Summary")
    st.write(results["founder_summary"])

    st.subheader("📊 VC Clustering")
    for cluster in results["clusters"]:
        st.markdown(f"**Cluster {cluster['cluster_id']}:** {cluster['description']}")
        st.markdown(", ".join(cluster["members"]))

    st.subheader("🧭 Visual Intelligence")
    for title, fig in results["visuals"].items():
        st.pyplot(fig)

    st.subheader("🤝 VC Co-Investment & Relationships")
    for r in results["relationships"]["co_investment"][:10]:
        st.markdown(f"- **{r['firm_a']}** and **{r['firm_b']}** → {r['type']} (Jaccard: {r['score']}, Cosine: {r['cosine_similarity']})")

    st.subheader("💡 Top VC Matches")
    for match in results["matches"]:
        st.markdown(f"- **{match['vc']}** | Score: {match['score']} | Cluster: {match['cluster']}")

    st.subheader("🔍 Similar Portfolio Companies")
    for item in results["similar_companies"]:
        st.markdown(f"- **{item['company_url']}** backed by: {', '.join(item['invested_vcs'])} (Score: {item['similarity_score']})")

    st.subheader("🚪 Strategic Gap Opportunities")
    for gap_item in results["gap"]:
        st.markdown(f"- Cluster {gap_item['cluster']} | Similarity: {gap_item['similarity']}")

    st.subheader("💬 Ask About Your Profile")
    user_question = st.text_input("Ask anything about your startup or the VC landscape...")
    if user_question:
        chatbot = ChatbotAgent(api_key=openai_api_key)  # Reinstantiating here ensures availability
        response = chatbot.create(results["vc_summaries"], results["founder_summary"])
        st.write(response)
