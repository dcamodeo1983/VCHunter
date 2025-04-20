import streamlit as st
import tempfile
import openai
from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator
from agents.vc_list_aggregator_agent import VCListAggregatorAgent
from agents.website_scraper_agent import VCWebsiteScraperAgent
from agents.portfolio_enricher_agent import PortfolioEnricherAgent
from agents.llm_embed_gap_match_chat import (
    LLMSummarizerAgent, EmbedderAgent, GapAnalysisAgent,
    FounderMatchAgent, ChatbotAgent
)
from agents.categorizer_agent import CategorizerAgent
from agents.relationship_agent import RelationshipAgent
from agents.visualization_agent import VisualizationAgent

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter - AI-Driven VC Landscape Analyzer")

uploaded_pitch = st.file_uploader("📄 Upload your one-pager (TXT or PDF)", type=["txt", "pdf"])
trigger_nvca = st.checkbox("Use default VC list (30 VCs)", value=True)
run_analysis = st.button("🚀 Run Full Analysis")

if "openai" in st.secrets:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    st.error("🔐 Please set your OpenAI API key in Streamlit Secrets.")
    st.stop()

if uploaded_pitch and run_analysis:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_pitch.read())
        pitch_path = tmp.name

    st.info("⏳ Running pipeline... this may take 1–3 minutes.")

    agents = {
        "nvca": VCListAggregatorAgent(),
        "scraper": VCWebsiteScraperAgent(),
        "portfolio": PortfolioEnricherAgent(),
        "summarizer": LLMSummarizerAgent(api_key=openai.api_key),
        "embedder": EmbedderAgent(api_key=openai.api_key),
        "categorizer": CategorizerAgent(api_key=openai.api_key),
        "relationship": RelationshipAgent,
        "matcher": FounderMatchAgent(),
        "gap": GapAnalysisAgent(),
        "chatbot": ChatbotAgent(api_key=openai.api_key),
        "visualizer": VisualizationAgent()
    }

    reader = FounderDocReaderAgent()
    founder_text = reader.extract_text(pitch_path)
    orchestrator = VCHunterOrchestrator(agents)
    results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca)

    st.success("✅ Analysis complete!")

    st.subheader("📌 Summary of Uploaded Startup")
    st.markdown(results["founder_summary"])

    st.subheader("📊 VC Landscape Visualizations")
    if results["visuals"]:
        st.pyplot(results["visuals"]["cluster_map"])
        st.pyplot(results["visuals"]["heatmap"])
        st.pyplot(results["visuals"]["network"])
    else:
        st.warning("No visualizations generated.")

    st.subheader("🔗 Relationship Overview")
    if results["relationships"]:
        for rel_type, data in results["relationships"].items():
            st.markdown(f"**{rel_type}** — {len(data)} relationships")
    else:
        st.warning("No relationship data found.")

    st.subheader("🎯 Best Matched VCs")
    if results["matches"]:
        for match in results["matches"]:
            st.markdown(f"- **{match['vc']}** — Similarity: {match['score']:.2f}, Cluster: {match['cluster']}")
    else:
        st.warning("No matches found.")

    st.subheader("🌌 White Space & Gaps")
    if results["gaps"]:
        for gap in results["gaps"]:
            st.markdown(f"- **Cluster {gap['label']}** — {gap['reason']}")
    else:
        st.warning("No gaps identified.")

    st.subheader("💬 Ask a Question")
    if results["chatbot"]:
        user_q = st.text_input("Ask your assistant something about this VC landscape:")
        if user_q:
            response = results["chatbot"].ask(user_q)
            st.markdown(f"**Answer:** {response}")
    else:
        st.warning("Chatbot not initialized.")
