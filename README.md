### Disaster Desk — Category‑Aware Disaster Q&A (RAG)

Ask questions over disaster guidelines with page‑level citations. Category filters (Flood/Earthquake/Cyclone/General), local embeddings by default, Streamlit UI.

### Features

End‑to‑end RAG: ingestion → embeddings → vector store → retrieval → answer.

Category filter via filename prefix.

Page‑level citations: [DocumentName, p.X].

Works offline (MiniLM) or with OpenAI/Google if keys are set.

Streamlit UI with Top‑K control and source previews.

### Quickstart

Python 3.10+ and uv installed

Create/activate venv:

uv venv venv

Terminal cmd: venv\Scripts\Activate

Install deps:

uv pip install -r requirements.txt

### Add PDFs

Put PDFs in data/raw_pdfs/

Name files with category prefix (case‑sensitive):

Flood_(name).pdf, Earthquake_(name).pdf, Cyclone_(name).pdf, General_(name).pdf

Example:

Flood_NDMA_Flood_Management.pdf

Flood_NDMA_Management_Urban_Flooding.pdf

Earthquake_NDMA_Management_of_Earthquakes.pdf

Cyclone_NDMA_Management_of_Cyclones.pdf

General_NDMA_Drought_Guidelines.pdf

General_NDMA_Tsunami_Guidelines.pdf

### .env configuration

OPENAI_API_KEY=

GOOGLE_API_KEY=

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

VECTORSTORE_DIR=./data/vectorstore

CHUNK_SIZE=1000

CHUNK_OVERLAP=150

TOP_K=4

SIMILARITY_CUTOFF=0.4


### Build the vector store

uv run python src/ingest.py --reset

Expect: “Persisted N chunks to ./data/vectorstore”


### Run the app

uv run streamlit run src/app.py

Open http://localhost:8501

Choose a category (or All), set Top‑K, ask a question, expand Sources for snippets.

### How it works (RAG)

Ingestion: PyMuPDF extracts per-page text → recursive chunking → embeddings → stored in Chroma with metadata (source, page, category).

Retrieval: get_relevant_docs queries Chroma for top‑k chunks, optionally filtered by category.

Augmentation: build_context concatenates retrieved chunks into the prompt context.

Generation: LLM (OpenAI/Google) answers strictly from the provided context; otherwise abstains with “Not enough info.” If no keys, a local extractive snippet is returned.

Citations: formatted as [DocumentName, p.X]; Sources expanders show previews.

### Sample queries

Flood: What are key preparedness measures for floods?

Cyclone: How should evacuation be managed before landfall?

Earthquake: What building safety measures are recommended?

General: What are minimum relief standards in shelters?

### Troubleshooting

“Not enough info.”: try Category=All and increase Top‑K to 6–8; ensure filenames start with Flood_/Earthquake_/Cyclone_/General_; re-ingest with --reset.

Blank text: some PDFs are scanned; PyMuPDF helps, but image‑only pages may need OCR.

First run is slow: local embedding model download can take 1–2 minutes.

### Project structure

data/
    raw_pdfs/
    vectorstore/
src/
    app.py
    ingest.py
    qa_chain.py
    utils.py
    config.py

Save the file and preview

Save README.md and preview it on GitHub or VS Code Markdown preview to ensure headings and lists render correctly.

### Screenshot of UI

![App UI](/UI Screenshot.png)

### Commit and push

git add README.md

git commit -m "Add project README"

git push