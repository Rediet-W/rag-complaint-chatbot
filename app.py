import streamlit as st
from src.retriever import ComplaintRetriever
from src.generator import ComplaintGenerator

# Initialize (cached so it doesn't reload on every click)
@st.cache_resource
def load_system():
    retriever = ComplaintRetriever()
    generator = ComplaintGenerator()
    return retriever, generator

retriever, generator = load_system()

st.title("CrediTrust Complaint Assistant")

# Chat input
question = st.text_input("Ask a question about customer complaints:")

if st.button("Submit"):
    if question:
        with st.spinner("Analyzing..."):
            context, sources = retriever.get_context(question)
            answer = generator.generate(context, question)
            
            st.subheader("Answer:")
            st.write(answer)
            
            with st.expander("View Retrieved Sources"):
                for s in sources:
                    st.write(f"**ID:** {s.metadata.get('id')} | **Text:** {s.page_content}")
    else:
        st.warning("Please enter a question.")