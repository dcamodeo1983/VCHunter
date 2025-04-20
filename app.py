import streamlit as st
from website_scraper_agent import WebsiteScraperAgent
from portfolio_enricher_agent import PortfolioEnricherAgent
from categorizer_agent import CategorizerAgent
from relationship_agent import RelationshipAgent
from visualization_agent import VisualizationAgent
from founder_doc_reader_and_orchestrator import FounderDocReaderAgent
from llm_embed_gap_match_chat import ChatbotAgent, FounderMatchAgent

import openai

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load data from the enrichment step
vc_to_companies = PortfolioEnricherAgent.load_company_map()
vc_to_vectors = PortfolioEnricherAgent.load_vector_map()

# Initialize all agents
scraper_agent = WebsiteScraperAgent()
enricher_agent = PortfolioEnricherAgent()
categorizer_agent = CategorizerAgent(api_key=openai.api_key)
relationship_agent = RelationshipAgent(vc_to_companies, vc_to_vectors)
visualizer_agent = VisualizationAgent()
reader_agent = FounderDocReaderAgent()
chatbot_agent = ChatbotAgent(api_key=openai.api_key)
matcher_agent = FounderMatchAgent()

# Your app logic starts here
st.title("VC Hunter App")

uploaded_file = st.file_uploader("Upload your founder doc", type=["pdf", "txt", "docx"])
if uploaded_file:
    with st.spinner("Analyzing..."):
        founder_data = reader_agent.read_and_summarize(uploaded_file)
        enriched =
