import os
import streamlit as st
from dotenv import load_dotenv

from llm_embed_gap_match_chat import ChatbotAgent, FounderMatchAgent, EmbedderAgent
from categorizer_agent import CategorizerAgent
from relationship_agent import RelationshipAgent
from visualization_agent import VisualizationAgent
from founder_doc_reader_and_orchestrator import FounderDocReaderAgent

# Load API key securely
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page config
st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🧠 VC Hunter App")
st.markdown("Upload your white paper and analyze venture capital relationships.")

uploaded_file = st.file_uploader("Upload Founder Document", type=["pdf", "txt", "docx"])

# Ensure both document and API key are present
if uploaded_file and openai_api_key:
    st.success("File uploaded successfully. Analyzing...")

    # Step 1: Summarize founder document
    reader_agent = FounderDocReaderAgent(api_key=openai_api_key)
    doc_summary = reader_agent.read_and_summarize(uploaded_file)

    st.subheader("📝 Summary")
    st.write(doc_summary)

    # Step 2: Categorize VC firms (embedding + clustering)
    categorizer_agent = CategorizerAgent(api_key=openai_api_key)
    vc_to_vectors, vc_to_companies = categorizer_agent.categorize_and_embed()
    cluster_map = categorizer_agent.cluster_map

    # PROCEED ONLY IF embedding and clustering are successful
    if vc_to_vectors and vc_to_companies and cluster_map:
        # Step 3: Analyze relationships
        relationship_agent = RelationshipAgent(vc_to_companies=vc_to_companies, vc_to_vectors=vc_to_vectors)
        relationship_graph = relationship_agent.analyze()

        # Step 4: Visualize relationships
        visualizer_agent = VisualizationAgent()
        visualizer_agent.render_graph(relationship_graph)

        # Step 5: Match founder to VCs
        st.subheader("💡 Match Insights")
        embedder_agent = EmbedderAgent(api_key=openai_api_key)
        founder_vec = embedder_agent.embed([doc_summary])[0]
        vc_names = list(vc_to_vectors.keys())
        vc_vecs = [vc_to_vectors[name] for name in vc_names]

        matcher_agent = FounderMatchAgent()
        match_result = matcher_agent.match(founder_vec, vc_vecs, vc_names, cluster_map)

        st.write(match_result)

        # Step 6: Chat interface
        st.subheader("💬 Chat with Your Startup Summary")
        chatbot_agent = ChatbotAgent(api_key=openai_api_key)
        user_question = st.text_input("Ask a question about your startup profile...")
        if user_question:
            reply = chatbot_agent.create(vc_summaries=[], founder_text=doc_summary)
            st.write(reply)
    else:
        st.error("Embedding or categorization failed. Please check VC data integrity.")
else:
    if not openai_api_key:
        st.warning("Missing OpenAI API Key.")
