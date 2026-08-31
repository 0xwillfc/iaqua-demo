import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_DIR = BASE_DIR / "data" / "corpus"
CATALOG_PATH = BASE_DIR / "data" / "catalogo.csv"
DB_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "aquia_rag"

# Provedores: "google" ou "nvidia"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google").lower()
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", EMBEDDING_PROVIDER).lower()

# --- Google ---
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"
GOOGLE_EMBEDDING_DIM = 768
GOOGLE_GENERATION_MODEL = "gemini-2.5-flash"

# --- NVIDIA (NIM, endpoint compatível com OpenAI) ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_EMBEDDING_MODEL = os.getenv(
    "NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"
)
NVIDIA_EMBEDDING_DIM = 1024
NVIDIA_GENERATION_MODEL = os.getenv(
    "NVIDIA_GENERATION_MODEL", "meta/llama-3.1-70b-instruct"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1024


def get_generation_model():
    return (
        NVIDIA_GENERATION_MODEL
        if GENERATION_PROVIDER == "nvidia"
        else GOOGLE_GENERATION_MODEL
    )
