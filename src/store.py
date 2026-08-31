import chromadb

from .config import COLLECTION_NAME, DB_DIR
from .embeddings import embed_texts


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks):
    col = get_collection()
    existing = set(col.get()["ids"]) if col.count() else set()
    novos = [c for c in chunks if c["id"] not in existing]
    if not novos:
        print("Nada novo a indexar.")
        return 0
    print(f"Gerando embeddings para {len(novos)} trechos...")
    vetores = embed_texts([c["texto"] for c in novos], "retrieval_document")
    col.add(
        ids=[c["id"] for c in novos],
        documents=[c["texto"] for c in novos],
        metadatas=[c["metadados"] for c in novos],
        embeddings=vetores,
    )
    return len(novos)


def search(query: str, top_k: int):
    col = get_collection()
    if col.count() == 0:
        raise RuntimeError(
            "Base vetorial vazia. Rode 'python -m src.ingest' primeiro."
        )
    qvec = embed_texts([query], "retrieval_query")[0]
    res = col.query(query_embeddings=[qvec], n_results=top_k)
    results = []
    ids = res.get("ids", [[]])[0] if res.get("ids") else [""] * len(res["documents"][0])
    for cid, doc, meta, dist in zip(
        ids, res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        results.append({"id": cid, "texto": doc, "metadados": meta, "distancia": dist})
    return results
