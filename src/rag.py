from .config import (
    GENERATION_PROVIDER,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    TOP_K,
    get_generation_model,
)
from .embeddings import google_client
from .store import search

PROMPT_SISTEMA = """Você é o AquIA, assistente técnico em piscicultura de \
tilápia (Oreochromis niloticus) e tambaqui (Colossoma macropomum).

REGRAS OBRIGATÓRIAS:
1. Responda EXCLUSIVAMENTE com base nos trechos fornecidos no CONTEXTO.
2. Cite a evidência usada no formato [Documento: <título>, p. <página>].
3. Se o contexto não contiver informação suficiente para responder, \
responda exatamente com: "Não há informação suficiente na base documental \
para responder a esta pergunta." e nada mais.
4. Se a pergunta contiver uma premissa falsa ou inconsistente com os trechos, \
aponte a premissa incorreta e corrija-a com base no contexto.
5. Não especule. Não use conhecimento externo ao contexto."""

PROMPT_USUARIO = """CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

Resposta:"""


def formatar_contexto(resultados):
    partes = []
    for i, r in enumerate(resultados, start=1):
        m = r["metadados"]
        cab = f"[{i}] {m['titulo']} ({m['instituicao']}"
        if m.get("ano"):
            cab += f", {m['ano']}"
        cab += f"), página {m['pagina']}"
        partes.append(f"{cab}\n{r['texto']}")
    return "\n\n".join(partes)


def gerar_resposta(pergunta: str, contexto: str) -> str:
    usuario = PROMPT_USUARIO.format(contexto=contexto, pergunta=pergunta)
    modelo = get_generation_model()

    if GENERATION_PROVIDER == "nvidia":
        import os

        from openai import OpenAI

        from .config import NVIDIA_BASE_URL

        client = OpenAI(
            base_url=NVIDIA_BASE_URL, api_key=os.environ["NVIDIA_API_KEY"]
        )
        resp = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": usuario},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return resp.choices[0].message.content

    client = google_client()
    response = client.models.generate_content(
        model=modelo,
        contents=usuario,
        config={
            "system_instruction": PROMPT_SISTEMA,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    return response.text


def responder(pergunta: str, top_k: int = None):
    top_k = top_k or TOP_K
    resultados = search(pergunta, top_k)
    contexto = formatar_contexto(resultados)
    resposta = gerar_resposta(pergunta, contexto)

    evidencias = [
        {
            "titulo": r["metadados"]["titulo"],
            "instituicao": r["metadados"].get("instituicao", ""),
            "ano": r["metadados"].get("ano", ""),
            "pagina": r["metadados"]["pagina"],
            "trecho": r["texto"][:300] + ("..." if len(r["texto"]) > 300 else ""),
            "similaridade": round(1 - r["distancia"], 4),
        }
        for r in resultados
    ]
    return {"resposta": resposta, "evidencias": evidencias}
