import streamlit as st
from config import CONFIG
from qa_chain import answer_query, get_relevant_docs

st.set_page_config(page_title="Topic Filter RAG", page_icon="📄", layout="wide")
st.title("Disaster Desk 📄")
st.caption("Ask questions over disaster docs. Answers are strictly from context with citations.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    category = st.selectbox("Category filter", ["All"] + list(CONFIG.categories))
    k = st.slider("Top-K (chunks)", 1, 10, CONFIG.top_k)
    similarity_cutoff = st.slider(
        "Similarity cutoff (visual only for now)",
        0.0, 1.0, CONFIG.similarity_cutoff, 0.05
    )

# Optional badge for active filter
if category != "All":
    st.caption(f"Filtering category: {category}")
else:
    st.caption("No category filter")

# Main UI
question = st.text_input("Enter your question", "")

if st.button("Answer") and question.strip():
    with st.spinner("Retrieving..."):
        cat = None if category == "All" else category
        result = answer_query(question.strip(), category=cat, k=k)

    # Answer block
    st.subheader("Answer")
    st.write(result["answer"])

    # Sources block with deduplicated expanders
    st.subheader("Sources")
    docs = get_relevant_docs(question.strip(), k=k, category=cat)

    seen = set()
    shown = 0
    for d in docs:
        label = f"{d.metadata.get('source', 'Document')}, p.{d.metadata.get('page', '1')}"
        if label in seen:
            continue
        seen.add(label)
        with st.expander(label):
            text = d.page_content or ""
            st.write(text[:800] + ("..." if len(text) > 800 else ""))
        shown += 1
        if shown >= 4:
            break

    if shown == 0:
        st.write("No sources or not enough info.")
