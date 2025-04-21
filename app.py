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

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup Streamlit
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter App")
st.markdown("Upload a white paper and analyze startup fit, VC clusters, and more.")

# Use session state for persistence
if "founder_docs" not in st.session_state:
    st.session_state["founder_docs"] = []
if "results" not in st.session_state:
    st.session_state["results"] = None

# Upload files
uploaded_files = st.file_uploader("📄 Upload Startup Concept Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    st.session_state["founder_docs"] = uploaded_files
    st.success(f"{len(uploaded_files)} document(s) uploaded.")

# Manual flag to confirm button press
run_analysis = st.button("🚀 Run Analysis")

# Run analysis only if files are in session AND button clicked AND key present
if run_analysis:
    if not st.session_state["founder_docs"]:
        st.warning("Please upload at least one document.")
    elif not openai_api_key:
        st.error("Missing OpenAI API key in environment.")
    else:
        progress_placeholder = st.empty()
        try:
            progress_placeholder.info("⏳ Initializing analysis agents...")

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

            full_text = ""
            for file in st.session_state["founder_docs"]:
                extracted = reader.extract_text(file)
                full_text += extracted + "\n"
                logger.info(f"Extracted {len(extracted)} characters.")

            progress_placeholder.info("🚀 Running full intelligence pipeline...")
            results = orchestrator.run(full_text)
            st.session_state["results"] = results
            progress_placeholder.success("✔️ Analysis complete.")

        except Exception as e:
            logger.exception("Analysis failed.")
            progress_placeholder.error(f"❌ Pipeline failed: {e}")
            st.stop()

# Display Results
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
        st.markdown(f"- **{item['company']}** ([Website]({item['url']})) backed by: {', '.join(item['vcs'])}")

    st.subheader("🚪 Strategic Gap Opportunities")
    for gap_item in results["gap"]:
        st.markdown(f"- Cluster {gap_item['cluster']} | Similarity: {gap_item['similarity']}")

    st.subheader("💬 Ask About Your Profile")
    user_question = st.text_input("Ask anything about your startup or the VC landscape...")
    if user_question:
        response = chatbot.create(results["vc_summaries"], results["founder_summary"])
        st.write(response)
