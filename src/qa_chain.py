# Question-answer chain with category filter and abstain rule
import os
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chat_models.base import BaseChatModel

from config import CONFIG

# ---- Embeddings / LLM selection ----
def get_embedder():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if os.getenv("GOOGLE_API_KEY"):
        return GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    return SentenceTransformerEmbeddings(model_name=CONFIG.embedding_model)

def get_llm() -> Optional[BaseChatModel]:
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return None  # local fallback handled later

# ---- Vector store / retriever ----
def get_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CONFIG.vectorstore_dir,
        collection_name="topic_filter_rag",
        embedding_function=get_embedder(),
    )

def get_relevant_docs(
    query: str,
    k: int = 4,
    category: Optional[str] = None,
) -> List[Document]:
    vs = get_vectorstore()
    filt = {"category": category} if category and category in CONFIG.categories else None
    search_kwargs = {"k": k}
    if filt:
        search_kwargs["filter"] = filt
    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    docs: List[Document] = retriever.invoke(query)
    return docs

# ---- Prompt and formatting ----
PROMPT = PromptTemplate.from_template(
    "You are a disaster response assistant. Answer only from the provided context.\n"
    "If the context is insufficient, reply exactly: Not enough info.\n"
    "Keep the answer concise (3-6 sentences).\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Cite sources as [DocumentName, p.X]."
)

def format_citations(docs: List[Document]) -> str:
    parts = []
    for d in docs[:4]:
        name = d.metadata.get("source", "Document")
        page = d.metadata.get("page", "1")
        parts.append(f"[{name}, p.{page}]")
    # dedupe, preserve order
    seen = set()
    uniq = []
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return " ".join(uniq[:4])

def build_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs[:4]:
        name = d.metadata.get("source", "Document")
        page = d.metadata.get("page", "1")
        blocks.append(f"Source: {name}, p.{page}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def answer_query(
    question: str,
    category: Optional[str] = None,
    k: int = 4,
) -> Dict[str, Any]:
    docs = get_relevant_docs(question, k=k, category=category)
    if not docs:
        return {"answer": "Not enough info.", "sources": []}

    context = build_context(docs)
    citations = format_citations(docs)

    llm = get_llm()
    if llm is None:
        # Local fallback: return an extractive snippet from the most relevant chunk
        text = docs[0].page_content.strip()
        snippet = text[:800]
        return {"answer": f"{snippet}\n\nSources: {citations}", "sources": citations}

    prompt = PROMPT.format(context=context, question=question)
    resp = llm.invoke(prompt)
    ans = resp.content.strip()
    if ans.lower().startswith("not enough info"):
        return {"answer": "Not enough info.", "sources": []}
    return {"answer": f"{ans}\n\nSources: {citations}", "sources": citations}

# Optional CLI for quick testing
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True, help="Question")
    p.add_argument("--cat", default=None, help="Category: Flood/Earthquake/Cyclone/General")
    p.add_argument("--k", type=int, default=CONFIG.top_k)
    args = p.parse_args()
    out = answer_query(args.q, category=args.cat, k=args.k)
    print(out["answer"])
