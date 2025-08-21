import os
import argparse
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils import pdf_to_pages
from config import CONFIG

RAW_DIR = "./data/raw_pdfs"

def infer_category(filename: str) -> str:
    head = os.path.basename(filename).split("_", 1)[0].strip().title()
    return head if head in CONFIG.categories else "General"

def get_embedder():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    if os.getenv("GOOGLE_API_KEY"):
        return GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    return SentenceTransformerEmbeddings(model_name=CONFIG.embedding_model)

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

def build_documents() -> List[Dict]:
    docs: List[Dict] = []
    if not os.path.isdir(RAW_DIR):
        print(f"Missing folder: {RAW_DIR}")
        return docs

    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]
    if not files:
        print(f"No PDFs found in {RAW_DIR}")
        return docs

    for fname in files:
        path = os.path.join(RAW_DIR, fname)
        category = infer_category(fname)
        pages = pdf_to_pages(path)
        any_text = False
        for page_num, text in pages:
            if not text:
                continue
            any_text = True
            for chunk in chunk_text(text):
                docs.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": os.path.basename(fname),
                            "page": page_num,
                            "category": category,
                        },
                    }
                )
        if not any_text:
            print(f"Warning: No extractable text in {fname} (consider OCR)")
    return docs

def persist(docs: List[Dict], reset: bool = False):
    if reset and os.path.isdir(CONFIG.vectorstore_dir):
        import shutil
        shutil.rmtree(CONFIG.vectorstore_dir)
    os.makedirs(CONFIG.vectorstore_dir, exist_ok=True)

    embedder = get_embedder()
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory=CONFIG.vectorstore_dir,
        collection_name="topic_filter_rag",
    )
    print(f"Persisted {len(texts)} chunks to {CONFIG.vectorstore_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete existing vector store before ingest")
    args = parser.parse_args()

    docs = build_documents()
    if not docs:
        print(f"No documents to ingest. Check {RAW_DIR} and file names.")
        return
    persist(docs, reset=args.reset)

if __name__ == "__main__":
    main()
