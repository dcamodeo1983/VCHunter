
import streamlit as st

st.set_page_config(page_title="VC Hunter", layout="wide")
st.title("🚀 VC Hunter - Founder Intelligence Explorer")

uploaded_file = st.file_uploader("Upload your one-pager (TXT or PDF)", type=["txt", "pdf"])
if uploaded_file:
    st.success("File uploaded! Full pipeline execution starts here.")
    st.write("→ Placeholder: embed, match, cluster, visualize")
else:
    st.info("Please upload a document to start your VC search.")
