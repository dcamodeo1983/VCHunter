import streamlit as st
import openai
import tempfile

from agents.vc_list_aggregator_agent import VCListAggregatorAgent
from agents.founder_doc_reader_and_orchestrator import FounderDocReaderAgent, VCHunterOrchestrator
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

# === Load OpenAI API key ===
if "openai" in st.secrets:
    openai.api_key = st.secrets["openai"]["api_key"]
else:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit → Settings → Secrets.")
    st.stop()

# === Streamlit UI ===
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter - Founder Intelligence Explorer")

uploaded_pitch = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["txt", "pdf"])
uploaded_csv = st.file_uploader("Upload optional CSV of VC URLs", type=["csv"])
run_pipeline = st.button("Run VC Intelligence Analysis")
trigger_nvca = st.checkbox("Enable GitHub VC List Scraping", value=True)

if uploaded_pitch and run_pipeline:
    with st.spinner("Running full analysis..."):

        # === Save uploaded pitch file ===
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_pitch.read())
            pitch_path = tmp_file.name

        # === Setup VC list aggregator ===
        vc_aggregator = VCListAggregatorAgent()
        if uploaded_csv is not None:
            with tempfile.NamedTemporaryFile(delete=False) as csv_tmp:
                csv_tmp.write(uploaded_csv.read())
                csv_path = csv_tmp.name
                vc_aggregator.add_csv_vcs(csv_path)

        # === Load and preview VC list ===
        vc_list = vc_aggregator.fetch_vc_records()
        st.subheader("🔍 VC List Loaded")
        st.write(f"Loaded {len(vc_list)} VC records")
        st.dataframe(vc_list[:5])

        # === Initialize all agents ===
        agents = {
            "nvca": vc_aggregator,
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

        # === Extract founder text ===
        reader = FounderDocReaderAgent()
        founder_text = reader.extract_text(pitch_path)
        st.subheader("📄 Extracted Founder Text")
        st.text(founder_text[:1000] or "⚠️ No text extracted!")

        # === Run pipeline ===
        orchestrator = VCHunterOrchestrator(agents)
        results = orchestrator.run(founder_text=founder_text, trigger_nvca=trigger_nvca)

        # === Show debug output ===
        st.success("✅ Analysis complete!")

        st.subheader("🧠 VC Summaries")
        if results["summaries"]:
            for summary in results["summaries"][:3]:
                st.json(summary)
        else:
            st.warning("⚠️ No summaries generated.")

        st.subheader("🔢 VC Embeddings & Clusters")
        if results["clusters"]:
            for cluster in results["clusters"]:
                st.markdown(f"**Cluster {cluster['cluster_id']}**: {cluster['description']}")
                st.write("Members:", cluster["members"])
        else:
            st.warning("⚠️ No clusters found.")

        st.subheader("🔗 Relationships")
        if results["relationships"]:
            st.write(results["relationships"])
        else:
            st.warning("⚠️ No relationships detected.")

        st.subheader("🎯 Top VC Matches")
        if results["matches"]:
            st.table(results["matches"])
        else:
            st.warning("⚠️ No matches returned.")

        st.subheader("🌌 White Space / Gap Analysis")
        if results["gaps"]:
            for gap in results["gaps"]:
                st.markdown(f"- **{gap['category']}** — Similarity: `{gap['score']:.3f}`")
                st.caption(gap["insight"])
        else:
            st.warning("⚠️ No gap insights generated.")
