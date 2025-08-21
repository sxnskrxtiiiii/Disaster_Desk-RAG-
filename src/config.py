# src/config.py
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppConfig:
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "./data/vectorstore")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    similarity_cutoff: float = float(os.getenv("SIMILARITY_CUTOFF", "0.4"))
    categories: tuple = ("Flood", "Earthquake", "Cyclone", "General")

CONFIG = AppConfig()
