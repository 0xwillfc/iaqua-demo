from .config import (
    EMBEDDING_PROVIDER,
    GOOGLE_EMBEDDING_DIM,
    GOOGLE_EMBEDDING_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_EMBEDDING_DIM,
    NVIDIA_EMBEDDING_MODEL,
)

if EMBEDDING_PROVIDER == "nvidia":
    import os

    from openai import OpenAI

    _client = None
    MODEL = NVIDIA_EMBEDDING_MODEL
    DIM = NVIDIA_EMBEDDING_DIM
else:
    from google import genai
    from google.genai import types

    _client = None
    MODEL = GOOGLE_EMBEDDING_MODEL
    DIM = GOOGLE_EMBEDDING_DIM


def _nvidia_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"]
        )
    return _client


def _google_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def google_client():
    return _google_client()


def embed_texts(texts, task_type="retrieval_document"):
    """Gera embeddings em lotes. Retorna lista de vetores."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "nvidia":
        input_type = "query" if task_type == "retrieval_query" else "passage"
        client = _nvidia_client()
        vectors = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(
                model=MODEL,
                input=batch,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(d.embedding for d in resp.data)
        return vectors
    else:
        client = _google_client()
        vectors = []
        batch_size = 90
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.models.embed_content(
                model=MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=DIM,
                ),
            )
            vectors.extend(e.values for e in response.embeddings)
        return vectors
