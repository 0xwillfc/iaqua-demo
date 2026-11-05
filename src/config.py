import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_DIR = BASE_DIR / "data" / "corpus"
CATALOG_PATH = BASE_DIR / "data" / "catalogo.csv"
DB_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "aquia_rag"

# Provedores de embedding: "google" ou "nvidia"
# Provedores de geracao: "google", "nvidia" ou "anthropic"
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

# --- Anthropic (Claude) - so geracao; a Anthropic nao oferece embeddings,
# entao a recuperacao continua no provedor de EMBEDDING_PROVIDER. ---
# claude-sonnet-5-5: bom equilibrio entre qualidade, velocidade e custo para a demo.
# Alternativa mais barata e rapida: claude-haiku-5-5.
ANTHROPIC_GENERATION_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
# low = respostas mais rapidas (bom para demo ao vivo); medium/high = mais raciocinio
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "low")
# O raciocinio interno do modelo tambem conta neste limite.
ANTHROPIC_MAX_TOKENS = 8000

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

TEMPERATURE = 0.1
MAX_OUTPUT_TOKENS = 1024


def get_generation_model():
    if GENERATION_PROVIDER == "anthropic":
        return ANTHROPIC_GENERATION_MODEL
    if GENERATION_PROVIDER == "nvidia":
        return NVIDIA_GENERATION_MODEL
    return GOOGLE_GENERATION_MODEL


def required_api_keys():
    """Variaveis de ambiente necessarias para os provedores configurados."""
    chaves = {"nvidia": "NVIDIA_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    necessarias = []
    for provedor in (EMBEDDING_PROVIDER, GENERATION_PROVIDER):
        chave = chaves.get(provedor)
        if chave and chave not in necessarias:
            necessarias.append(chave)
    return necessarias
