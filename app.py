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
run_pipeline = st.button("Run VC Intelligence Analysis")

if uploaded_pitch and run_pipeline:
    with st.spinner("Running full analysis..."):

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_pitch.read())
            pitch_path = tmp_file.name

        vc_aggregator = VCListAggregatorAgent()
        vc_list = vc_aggregator.fetch_vc_records()

        st.subheader("🔍 VC List Used")
        st.write(f"Loaded {len(vc_list)} VC records")
        st.dataframe(vc_list)

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

        reader = FounderDocReaderAgent()
        founder_text = reader.extract_text(pitch_path)
        st.subheader("📄 Extracted Founder Text")
        st.text(founder_text[:1000] or "⚠️ No text extracted!")

        st.info("🧠 Running Orchestrator...")
        orchestrator = VCHunterOrchestrator(agents)
        results = orchestrator.run(founder_text=founder_text, trigger_nvca=True)

        st.success("✅ Analysis complete!")

        st.subheader("🧠 VC Summaries")
        if results["summaries"]:
            for i, summary in enumerate(results["summaries"]):
                st.markdown(f"**Summary {i+1}**")
                st.text(summary[:300])
        else:
            st.warning("⚠️ No summaries generated.")

        st.subheader("🧠 Clusters")
        if results["clusters"]:
            for c in results["clusters"]:
                st.write(c)
        else:
            st.warning("⚠️ No clusters returned.")

        st.subheader("🔗 Relationships")
        st.write(results["relationships"] or "⚠️ No relationships found.")

        st.subheader("🎯 Matches")
        st.write(results["matches"] or "⚠️ No matches found.")

        st.subheader("🌌 Gaps")
        st.write(results["gaps"] or "⚠️ No white space detected.")
