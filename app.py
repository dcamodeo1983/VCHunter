import os
import streamlit as st
from dotenv import load_dotenv

# Updated imports from the 'agents' package
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

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter App")
st.markdown("Upload your white paper to analyze startup fit, VC categories, co-investment networks, and portfolio signals.")

# 🔁 Persistent state
if "founder_docs" not in st.session_state:
    st.session_state["founder_docs"] = []

uploaded_file = st.file_uploader("📄 Upload Your Startup Concept (PDF, TXT, or DOCX)", type=["pdf", "txt", "docx"])
if uploaded_file:
    st.session_state["founder_docs"].append(uploaded_file)
    st.success(f"✅ Added: {uploaded_file.name}")

if st.button("🚀 Run Full Analysis") and st.session_state["founder_docs"]:
    try:
        st.info("Running full intelligence pipeline...")

        # Agent setup
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
        for doc in st.session_state["founder_docs"]:
            full_text += "\n" + reader.extract_text(doc)

        results = orchestrator.run(full_text)

        st.success("✔️ Analysis complete.")

        # 📝 Founder Summary
        st.subheader("📝 Founder Summary")
        st.write(results["founder_summary"])

        # 🧠 VC Clusters
        st.subheader("📊 VC Clustering")
        for cluster in results["clusters"]:
            st.markdown(f"**Cluster {cluster['cluster_id']}:** {cluster['description']}")
            st.markdown(", ".join(cluster['members"]))

        # 🔥 Visuals
        st.subheader("🧭 Visual Intelligence")
        for title, fig in results["visuals"].items():
            st.pyplot(fig)

        # 🤝 Relationships
        st.subheader("🤝 VC Co-Investment & Relationships")
        for r in results["relationships"]["co_investment"][:10]:
            st.markdown(f"- **{r['firm_a']}** and **{r['firm_b']}** → {r['type']} (Jaccard: {r['score']}, Cosine: {r['cosine_similarity']})")

        # 💡 Match Insights
        st.subheader("💡 Top VC Matches")
        for match in results["matches"]:
            st.markdown(f"- **{match['vc']}** | Score: {match['score']} | Cluster: {match['cluster']}")

        # 🕵️ Similar Companies
        st.subheader("🔍 Similar Portfolio Companies")
        for item in results["similar_companies"]:
            st.markdown(f"- **{item['company']}** ([Website]({item['url']})) backed by: {', '.join(item['vcs'])}")

        # 🧠 Gap Signals
        st.subheader("🚪 Strategic Gap Opportunities")
        for gap_item in results["gap"]:
            st.markdown(f"- Cluster {gap_item['cluster']} | Similarity: {gap_item['similarity']}")

        # 💬 Chat with your summary
        st.subheader("💬 Ask About Your Profile")
        user_question = st.text_input("Ask anything about your startup or the VC landscape...")
        if user_question:
            response = chatbot.create(results["vc_summaries"], results["founder_summary"])
            st.write(response)

    except Exception as e:
        st.error(f"❌ Pipeline execution failed: {e}")
else:
    st.warning("Please upload at least one document to enable analysis.")
