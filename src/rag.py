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


def formatar_evidencias(resultados):
    return [
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


def buscar(pergunta: str, top_k: int = None):
    """Etapa de recuperacao: devolve (contexto para o modelo, evidencias)."""
    resultados = search(pergunta, top_k or TOP_K)
    return formatar_contexto(resultados), formatar_evidencias(resultados)


_anthropic_client = None


def _parametros_anthropic(modelo: str, usuario: str) -> dict:
    global _anthropic_client
    import anthropic

    from .config import ANTHROPIC_EFFORT, ANTHROPIC_MAX_TOKENS

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # le ANTHROPIC_API_KEY

    # Sem temperature: os modelos Claude atuais nao aceitam parametros de
    # amostragem; o raciocinio e controlado por output_config.effort.
    # fallbacks="default": se o filtro de seguranca recusar por engano,
    # a API refaz a chamada em outro modelo em vez de devolver erro.
    return dict(
        model=modelo,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": usuario}],
        output_config={"effort": ANTHROPIC_EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def _aviso_parada(stop_reason: str) -> str:
    if stop_reason == "refusal":
        return "O modelo recusou responder a esta pergunta."
    if stop_reason == "max_tokens":
        return "\n\n_(resposta interrompida pelo limite de tamanho)_"
    return ""


def _anthropic_client_messages():
    # _parametros_anthropic cria o cliente; chamado antes deste acesso.
    return _anthropic_client.beta.messages


def gerar_resposta(pergunta: str, contexto: str) -> str:
    usuario = PROMPT_USUARIO.format(contexto=contexto, pergunta=pergunta)
    modelo = get_generation_model()

    if GENERATION_PROVIDER == "anthropic":
        params = _parametros_anthropic(modelo, usuario)
        resp = _anthropic_client_messages().create(**params)
        if resp.stop_reason == "refusal":
            return _aviso_parada("refusal")
        texto = "".join(b.text for b in resp.content if b.type == "text").strip()
        return texto + _aviso_parada(resp.stop_reason)

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


def gerar_resposta_stream(pergunta: str, contexto: str):
    """Gera a resposta em pedacos (para exibir enquanto o modelo escreve).

    So o provedor anthropic transmite de fato; os demais devolvem o texto
    inteiro de uma vez.
    """
    if GENERATION_PROVIDER != "anthropic":
        yield gerar_resposta(pergunta, contexto)
        return

    usuario = PROMPT_USUARIO.format(contexto=contexto, pergunta=pergunta)
    params = _parametros_anthropic(get_generation_model(), usuario)
    with _anthropic_client_messages().stream(**params) as stream:
        for texto in stream.text_stream:
            yield texto
        final = stream.get_final_message()
    aviso = _aviso_parada(final.stop_reason)
    if aviso:
        yield aviso if aviso.startswith("\n") else "\n\n" + aviso


def responder(pergunta: str, top_k: int = None):
    contexto, evidencias = buscar(pergunta, top_k)
    resposta = gerar_resposta(pergunta, contexto)
    return {"resposta": resposta, "evidencias": evidencias}
