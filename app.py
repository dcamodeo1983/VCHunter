import os
import streamlit as st
from llm_embed_gap_match_chat import ChatbotAgent, FounderMatchAgent
from categorizer_agent import CategorizerAgent
from relationship_agent import RelationshipAgent
from visualization_agent import VisualizationAgent
from founder_doc_reader_and_orchestrator import FounderDocReaderAgent

openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="VC Hunter", layout="wide")

st.title("🧠 VC Hunter App")
st.markdown("Upload your white paper and analyze venture capital relationships.")

uploaded_file = st.file_uploader("Upload Founder Document", type=["pdf", "txt", "docx"])

if uploaded_file and openai_api_key:
    st.success("File uploaded successfully. Analyzing...")
    
    # Instantiate agents
    reader_agent = FounderDocReaderAgent(api_key=openai_api_key)
    doc_summary = reader_agent.read_and_summarize(uploaded_file)

    st.subheader("📝 Summary")
    st.write(doc_summary)

    categorizer_agent = CategorizerAgent(api_key=openai_api_key)
    vc_to_vectors, vc_to_companies = categorizer_agent.categorize_and_embed()

    relationship_agent = RelationshipAgent(vc_to_companies, vc_to_vectors)
    relationship_graph = relationship_agent.build_relationship_graph()

    visualizer_agent = VisualizationAgent()
    visualizer_agent.render_graph(relationship_graph)

    st.subheader("💡 Match Insights")
    matcher_agent = FounderMatchAgent()
    match_result = matcher_agent.match(doc_summary, vc_to_vectors)

    st.write(match_result)

    st.subheader("💬 Chat with Your Startup Summary")
    chatbot_agent = ChatbotAgent(api_key=openai_api_key)
    user_question = st.text_input("Ask a question about your startup profile...")
    if user_question:
        reply = chatbot_agent.chat(user_question, doc_summary)
        st.write(reply)
