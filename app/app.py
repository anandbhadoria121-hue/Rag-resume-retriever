"""Streamlit frontend for the resume RAG pipeline.

Run with:
    streamlit run app.py

Requires rag_pipeline.py in the same folder and the .env with
PINECONE_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY.
"""

import streamlit as st

from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Resume Search", page_icon="📄", layout="wide")


# Load the pipeline ONCE and reuse it across reruns.
# Without this, the embedding model reloads on every button click.
@st.cache_resource(show_spinner="Loading embedding model and connecting to Pinecone...")
def get_pipeline():
    return RAGPipeline(index_name="resumes")


rag = get_pipeline()

# ---------- Sidebar: settings ----------
with st.sidebar:
    st.header("Settings")

    llm = st.radio(
        "Answer with",
        options=["gemini", "llama"],
        format_func=lambda x: "Gemini 2.5 Flash" if x == "gemini" else "Llama 3.3 70B (OpenRouter)",
    )

    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=15, value=5)

    job_class = st.text_input(
        "Filter by job class (optional)",
        placeholder="e.g. ACCOUNTANT",
        help="Matches the folder name the resume came from. Leave empty to search all.",
    ).strip() or None

# ---------- Main: query ----------
st.title("Resume Search")
st.caption("Ask a question — answers are grounded in the resumes indexed in Pinecone.")

query = st.text_input(
    "Your question",
    placeholder="e.g. Which candidates have experience with financial reporting?",
)

if st.button("Search", type="primary", disabled=not query):
    # 1) Retrieve
    with st.spinner("Retrieving relevant chunks..."):
        matches = rag.retrieve(query, top_k=top_k, job_class=job_class)

    if not matches:
        st.warning("No relevant chunks found. Try rephrasing or removing the class filter.")
        st.stop()

    # 2) Generate
    context = rag.build_context(matches)
    with st.spinner(f"Generating answer with {llm}..."):
        try:
            if llm == "gemini":
                answer = rag.generate_gemini(query, context)
            else:
                answer = rag.generate_llama(query, context)
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    # 3) Display
    st.subheader("Answer")
    st.write(answer)

    sources = sorted({m["metadata"]["filename"] for m in matches})
    st.subheader("Sources")
    st.write(", ".join(sources))

    with st.expander(f"Retrieved chunks ({len(matches)})"):
        for m in matches:
            meta = m["metadata"]
            st.markdown(
                f"**{meta['filename']}** · class: `{meta['class']}` "
                f"· score: `{m['score']:.3f}`"
            )
            st.text(meta["text"][:1500])
            st.divider()