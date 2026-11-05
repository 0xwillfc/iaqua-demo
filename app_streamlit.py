"""Interface web do IAqua-RAG para demonstracao ao vivo (Streamlit Community Cloud).

Roda sobre o mesmo src/rag.py do projeto original, apontando para um
data/chroma_db ja indexado (nao requer os 14 PDFs nem reingestao).
"""
import base64
import html
import os
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud expoe secrets via st.secrets, nao como variavel de ambiente.
# src/config.py le tudo com os.getenv() na importacao, entao a ponte abaixo
# precisa rodar antes dele. Os secrets tem prioridade sobre o ambiente.
# Atencao: depois de mudar os secrets no Streamlit Cloud, reinicie o app
# (menu ⋮ -> Reboot app) para o config ser lido de novo.
try:
    for _chave, _valor in st.secrets.items():
        if isinstance(_valor, (str, int, float, bool)):
            os.environ[_chave] = str(_valor)
except Exception:
    pass  # sem secrets.toml local (rodando so com .env) - segue normal

from src.config import get_generation_model, required_api_keys  # noqa: E402
from src.rag import buscar, gerar_resposta_stream  # noqa: E402

BASE = Path(__file__).resolve().parent
LOGO = BASE / "assets" / "iaqua-rag-logo.svg"

# Modelo usado no estudo (banner/artigo) - a demo ao vivo usa outro gerador.
MODELO_DO_ESTUDO = "Llama 3.1-70B-Instruct"
NOMES_MODELOS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}
RESPOSTA_RECUSA = "Não há informação suficiente na base documental"

EXEMPLOS = [
    (
        "Qualidade da água",
        "Quais variáveis de qualidade da água devem ser monitoradas diariamente na "
        "produção de tilápia em tanques-rede?",
    ),
    (
        "CONAMA classe 2",
        "Quais parâmetros de qualidade da água são classificados como padrões CONAMA "
        "classe 2 relevantes para a piscicultura e qual deles também é monitorado "
        "diariamente nas boas práticas de manejo?",
    ),
    (
        "Premissa falsa",
        "Como as tilápias regulam a própria temperatura corporal para manter o "
        "consumo de ração estável ao longo do ano?",
    ),
    (
        "Fora da base",
        "Qual é o preço atual do quilo do filé de tilápia no mercado brasileiro?",
    ),
]

st.set_page_config(
    page_title="IAqua-RAG",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Manrope:wght@700;800&display=swap');
:root { --navy:#0b2f5b; --aqua:#12a5dc; --mist:#edf6fa; --line:#d9e6ec; --ink:#16333d; --muted:#60747b; }
.stApp { background:linear-gradient(180deg,#f3f9fb 0%,#ffffff 40%); color:var(--ink); }
html, body, .stApp, .stMarkdown, button, textarea { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--navy) !important; }
.block-container { max-width:1240px; padding-top:1.4rem; padding-bottom:2.5rem; }
header[data-testid="stHeader"] { background:transparent; }

.hero { display:flex; align-items:center; gap:30px; flex-wrap:wrap; padding-bottom:18px; margin-bottom:20px; border-bottom:1px solid var(--line); }
.hero img { height:92px; width:auto; }
.hero p { flex:1; min-width:260px; margin:0; color:var(--muted); font-size:1.1rem; line-height:1.5; }
.hero p b { color:var(--navy); }

.eyebrow { color:var(--aqua); text-transform:uppercase; letter-spacing:.12em; font-size:.7rem; font-weight:800; margin:2px 0 6px; }
.st-key-consulta [data-testid="stForm"] { border:0; padding:0; }
[data-testid="stTextArea"] textarea { border-radius:13px; border-color:#cfe0e8; font-size:1.02rem; }
.stButton>button, [data-testid="stFormSubmitButton"]>button { border-radius:11px; font-weight:700; }
[data-testid="stFormSubmitButton"]>button { background:var(--navy); border:0; color:#fff; min-height:44px; }
[data-testid="stFormSubmitButton"]>button:hover { background:#0a4f8c; color:#fff; }
.stButton>button[kind="secondary"] { background:#fff; border:1px solid var(--line); color:var(--navy); font-size:.84rem; }
.stButton>button[kind="secondary"]:hover { border-color:var(--aqua); color:var(--aqua); }
.dica { color:#8aa0a7; font-size:.74rem; margin:-4px 0 0; text-align:right; }

.st-key-resposta { background:#fff; border:1px solid var(--line); border-left:5px solid var(--aqua); border-radius:18px; padding:20px 24px; box-shadow:0 12px 32px rgba(11,47,91,.06); }
.st-key-resposta p, .st-key-resposta li { font-size:1.04rem; line-height:1.65; color:#183942; }
.st-key-resposta-recusa { background:#fff8e7; border:1px solid #f0d99c; border-left:5px solid #e0a82e; border-radius:18px; padding:20px 24px; }
.st-key-resposta-recusa p { font-size:1.04rem; line-height:1.6; color:#5c4710; }
.tag-recusa { display:inline-block; background:#fbe7b5; color:#7a5800; border-radius:999px; padding:3px 10px; font-size:.72rem; font-weight:800; margin-bottom:6px; }
.meta-resposta { color:#7d9399; font-size:.78rem; margin-top:8px; }

.evidence-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:15px 17px; margin-bottom:11px; box-shadow:0 8px 22px rgba(11,47,91,.045); }
.evidence-card.citada { border-color:#9fd8ee; box-shadow:0 0 0 2px #d4effa, 0 8px 22px rgba(11,47,91,.06); }
.evidence-card.apagada { opacity:.62; }
.source-line { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; margin-bottom:8px; }
.source-title { font-weight:700; color:var(--navy); font-size:.92rem; line-height:1.3; }
.source-meta { color:var(--muted); font-size:.76rem; }
.pills { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.page-pill { white-space:nowrap; background:var(--mist); color:var(--navy); border-radius:999px; padding:4px 10px; font-size:.72rem; font-weight:800; }
.cit-pill { white-space:nowrap; background:var(--aqua); color:#fff; border-radius:999px; padding:4px 10px; font-size:.68rem; font-weight:800; letter-spacing:.04em; }
.quote { color:#526970; font-size:.88rem; line-height:1.5; border-left:2px solid #a9d9ec; padding-left:12px; }
.score { display:flex; align-items:center; gap:8px; color:#759097; font-size:.72rem; margin-top:9px; }
.bar { flex:1; max-width:140px; height:5px; background:#e6eff3; border-radius:99px; overflow:hidden; }
.bar span { display:block; height:100%; background:linear-gradient(90deg,#0a4f8c,#12a5dc); }
.empty { border:1.5px dashed #cfe0e8; border-radius:16px; padding:30px 26px; color:var(--muted); text-align:center; font-size:.93rem; line-height:1.55; }
.legenda { color:#7d9399; font-size:.78rem; margin:-2px 0 10px; }

.footer-note { color:#6f878e; text-align:center; font-size:.8rem; line-height:1.5; margin-top:34px; padding-top:14px; border-top:1px solid var(--line); }
.footer-note b { color:var(--navy); }
@media (max-width:800px) { .hero img { height:68px; } .hero { gap:14px; } }
</style>
"""


def nome_modelo() -> str:
    modelo = get_generation_model()
    return NOMES_MODELOS.get(modelo, modelo)


def logo_html() -> str:
    dados = base64.b64encode(LOGO.read_bytes()).decode()
    return f'<img src="data:image/svg+xml;base64,{dados}" alt="IAqua-RAG">'


@st.cache_resource
def cache_respostas() -> dict:
    # Compartilhado entre visitantes: pergunta repetida (ex.: exemplos)
    # responde na hora e sem custo de API.
    return {}


def usar_exemplo(texto: str) -> None:
    st.session_state.pergunta = texto
    st.session_state.executar = True


def citacoes(resposta: str) -> list:
    return re.findall(r"\[Documento:[^\]]*\]", resposta)


def foi_citada(ev: dict, cits: list) -> bool:
    titulo = str(ev["titulo"]).strip().lower()
    pagina = re.escape(str(ev["pagina"]))
    return any(
        titulo in c.lower() and re.search(rf"p\.\s*{pagina}\b", c) for c in cits
    )


def mensagem_erro(e: Exception) -> str:
    texto = str(e)
    if "end of life" in texto or "Error code: 410" in texto:
        return (
            f"O modelo de geração configurado (**{get_generation_model()}**) foi "
            "desativado pelo provedor. Nos secrets do Streamlit, use "
            '`GENERATION_PROVIDER = "anthropic"` e reinicie o app (menu ⋮ → '
            "**Reboot app**)."
        )
    if "authentication" in texto.lower() or "Error code: 401" in texto:
        return "A chave de API foi recusada. Confira as chaves nos secrets do app."
    return f"Não foi possível consultar o sistema agora. Detalhe técnico: {texto}"


def render_evidencias(evidencias, resposta: str | None = None) -> None:
    cits = citacoes(resposta) if resposta else []
    algum_citado = any(foi_citada(ev, cits) for ev in evidencias)
    if algum_citado:
        st.markdown(
            '<div class="legenda">Em destaque: trechos citados na resposta.</div>',
            unsafe_allow_html=True,
        )
    for i, ev in enumerate(evidencias, start=1):
        citada = foi_citada(ev, cits)
        classe = "citada" if citada else ("apagada" if algum_citado else "")
        meta = ", ".join(
            str(x)
            for x in (ev["instituicao"], ev["ano"])
            if x and str(x).strip().lower() not in ("não informado", "nao informado")
        )
        relevancia = max(0.0, min(1.0, float(ev["similaridade"])))
        pill_cit = '<span class="cit-pill">CITADO</span>' if citada else ""
        st.markdown(
            f"""
            <div class="evidence-card {classe}">
              <div class="source-line">
                <div>
                  <div class="source-title">[{i}] {html.escape(str(ev["titulo"]))}</div>
                  <div class="source-meta">{html.escape(meta)}</div>
                </div>
                <div class="pills">{pill_cit}<span class="page-pill">p. {html.escape(str(ev["pagina"]))}</span></div>
              </div>
              <div class="quote">“{html.escape(ev["trecho"])}”</div>
              <div class="score">similaridade {relevancia:.2f}
                <div class="bar"><span style="width:{relevancia * 100:.0f}%"></span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_resposta(resultado: dict) -> None:
    resposta = resultado["resposta"]
    meta = (
        f'<div class="meta-resposta">Gerada por {html.escape(resultado["modelo"])} '
        f'a partir de {len(resultado["evidencias"])} trechos recuperados'
        + (f' · {resultado["segundos"]:.0f} s' if resultado.get("segundos") else "")
        + "</div>"
    )
    if resposta.strip().startswith(RESPOSTA_RECUSA):
        with st.container(key="resposta-recusa"):
            st.markdown(
                '<span class="tag-recusa">SEM EVIDÊNCIA NA BASE</span>',
                unsafe_allow_html=True,
            )
            st.markdown(resposta)
            st.caption("O sistema recusa em vez de inventar uma resposta.")
    else:
        with st.container(key="resposta"):
            st.markdown(resposta)
            st.markdown(meta, unsafe_allow_html=True)


# ---------------------------------------------------------------- pagina
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="hero">
      {logo_html()}
      <p>Consulta técnica em piscicultura de <b>tilápia</b> e <b>tambaqui</b>.
      As respostas são fundamentadas em documentos técnicos e citam
      documento e página de onde a informação foi tirada.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

esquerda, direita = st.columns([1.15, 1], gap="large")

with esquerda:
    st.markdown('<div class="eyebrow">Sua pergunta</div>', unsafe_allow_html=True)
    with st.container(key="consulta"):
        with st.form("form-consulta", border=False):
            st.text_area(
                "Pergunta",
                key="pergunta",
                height=104,
                label_visibility="collapsed",
                placeholder="Ex.: Qual a temperatura ideal para criação de tilápia em tanques-rede?",
            )
            enviado = st.form_submit_button("Consultar", use_container_width=True)
    st.markdown('<p class="dica">Ctrl + Enter também consulta</p>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:6px">Ou experimente</div>', unsafe_allow_html=True)
    colunas = st.columns(2) + st.columns(2)
    for coluna, (rotulo, texto) in zip(colunas, EXEMPLOS):
        coluna.button(
            rotulo,
            key=f"ex-{rotulo}",
            on_click=usar_exemplo,
            args=(texto,),
            use_container_width=True,
        )
    area_resposta = st.container()

with direita:
    st.markdown('<div class="eyebrow">Evidências recuperadas</div>', unsafe_allow_html=True)
    area_evidencias = st.container()

# ---------------------------------------------------------------- consulta
executar = enviado or st.session_state.pop("executar", False)
if executar:
    pergunta = (st.session_state.get("pergunta") or "").strip()
    faltando = [k for k in required_api_keys() if not os.environ.get(k)]
    cache = cache_respostas()
    if not pergunta:
        area_resposta.warning("Digite uma pergunta.")
    elif faltando:
        area_resposta.error(
            "Configuração incompleta: "
            + ", ".join(faltando)
            + " não definida(s). Configure em Settings → Secrets e reinicie o app."
        )
    elif pergunta in cache:
        st.session_state.resultado = cache[pergunta]
    else:
        inicio = time.time()
        try:
            with area_resposta:
                with st.status("Buscando nos documentos técnicos...", expanded=False) as status:
                    contexto, evidencias = buscar(pergunta)
                    status.update(
                        label=f"{len(evidencias)} trechos relevantes encontrados",
                        state="complete",
                    )
            with area_evidencias:
                render_evidencias(evidencias)
            with area_resposta:
                st.markdown('<div class="eyebrow" style="margin-top:14px">Resposta</div>', unsafe_allow_html=True)
                with st.spinner(f"{nome_modelo()} está redigindo a resposta..."):
                    with st.container(key="resposta"):
                        resposta = st.write_stream(gerar_resposta_stream(pergunta, contexto))
            resultado = {
                "pergunta": pergunta,
                "resposta": resposta if isinstance(resposta, str) else "".join(map(str, resposta)),
                "evidencias": evidencias,
                "modelo": nome_modelo(),
                "segundos": time.time() - inicio,
            }
            cache[pergunta] = resultado
            st.session_state.resultado = resultado
            st.rerun()  # redesenha com os trechos citados em destaque
        except Exception as e:  # nunca deixar a demo travar em tela de erro cru
            st.session_state.resultado = None
            area_resposta.error(mensagem_erro(e))

# ---------------------------------------------------------------- resultado
resultado = st.session_state.get("resultado")
if resultado:
    with area_resposta:
        st.markdown('<div class="eyebrow" style="margin-top:14px">Resposta</div>', unsafe_allow_html=True)
        render_resposta(resultado)
    with area_evidencias:
        render_evidencias(resultado["evidencias"], resultado["resposta"])
elif not executar:
    area_evidencias.markdown(
        '<div class="empty">Faça uma pergunta ou escolha um exemplo.<br>'
        "Os trechos dos documentos usados na resposta aparecem aqui, com "
        "documento, página e similaridade.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="footer-note">
      Demonstração ao vivo: geração das respostas com <b>{html.escape(nome_modelo())}</b>.
      Os resultados do estudo apresentados no banner foram obtidos com
      <b>{MODELO_DO_ESTUDO}</b>.<br>
      Protótipo de pesquisa. Confira sempre a fonte citada antes de usar a resposta.
    </div>
    """,
    unsafe_allow_html=True,
)
