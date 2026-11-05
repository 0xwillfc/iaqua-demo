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

    if GENERATION_PROVIDER == "anthropic":
        return _gerar_anthropic(modelo, usuario)

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


_anthropic_client = None


def _gerar_anthropic(modelo: str, usuario: str) -> str:
    global _anthropic_client
    import anthropic

    from .config import ANTHROPIC_EFFORT, ANTHROPIC_MAX_TOKENS

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # le ANTHROPIC_API_KEY

    # Sem temperature: os modelos Claude atuais nao aceitam parametros de
    # amostragem; o raciocinio e controlado por output_config.effort.
    # fallbacks="default": se o filtro de seguranca recusar por engano,
    # a API refaz a chamada em outro modelo em vez de devolver erro.
    resp = _anthropic_client.beta.messages.create(
        model=modelo,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": usuario}],
        output_config={"effort": ANTHROPIC_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        return "O modelo recusou responder a esta pergunta."
    texto = "".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "max_tokens":
        texto += "\n\n_(resposta interrompida pelo limite de tamanho)_"
    return texto


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
