"""Interface web do IAqua-RAG para demonstracao ao vivo (Streamlit Community Cloud).

Roda sobre o mesmo src/rag.py do projeto original, apontando para um
data/chroma_db ja indexado (nao requer os 14 PDFs nem reingestao).
"""
import base64
import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud expoe secrets via st.secrets, nao como variavel de ambiente.
# src/config.py le tudo com os.getenv(), entao a ponte abaixo e obrigatoria -
# sem ela, EMBEDDING_PROVIDER cai no default "google" e quebra o import.
try:
    for _chave, _valor in st.secrets.items():
        os.environ.setdefault(_chave, str(_valor))
except Exception:
    pass  # sem secrets.toml local (rodando so com .env) - segue normal

from src.config import get_generation_model, required_api_keys  # noqa: E402
from src.rag import responder  # noqa: E402

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
:root { --navy:#0b2f5b; --teal:#087f78; --aqua:#12a5dc; --mist:#edf6fa; --line:#d9e6ec; --ink:#16333d; --muted:#60747b; }
.stApp { background:linear-gradient(180deg,#f3f9fb 0%,#ffffff 46%); color:var(--ink); }
html, body, .stApp, .stMarkdown, button, textarea { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.02em; color:var(--navy) !important; }
.block-container { max-width:1240px; padding-top:1.6rem; padding-bottom:2.5rem; }
header[data-testid="stHeader"] { background:transparent; }

.hero { display:flex; align-items:center; gap:28px; flex-wrap:wrap; margin-bottom:.4rem; }
.hero img { height:96px; width:auto; }
.hero-text { flex:1; min-width:260px; }
.subtitle { color:var(--muted); font-size:1.08rem; line-height:1.45; margin:0 0 .7rem; }
.badges { display:flex; gap:8px; flex-wrap:wrap; }
.badge { border:1px solid var(--line); background:#fff; border-radius:999px; padding:5px 11px; font-size:.74rem; font-weight:700; color:var(--navy); letter-spacing:.03em; }
.badge.live { background:#e6f7ef; border-color:#b7e5cd; color:#11744a; }

.pipeline { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:18px 0 22px; }
.pipe-step { background:var(--mist); border:1px solid #d3e8f1; border-radius:12px; padding:10px 8px; text-align:center; color:var(--navy); font-size:.8rem; font-weight:700; }
.pipe-step b { color:var(--aqua); margin-right:4px; }

.eyebrow { color:var(--aqua); text-transform:uppercase; letter-spacing:.12em; font-size:.7rem; font-weight:800; margin:4px 0 6px; }
[data-testid="stTextArea"] textarea { border-radius:13px; border-color:#cfe0e8; font-size:1rem; }
.stButton>button { border-radius:11px; font-weight:700; }
.stButton>button[kind="primary"] { background:var(--navy); border:0; color:#fff; }
.stButton>button[kind="primary"]:hover { background:#0a4f8c; color:#fff; }
.stButton>button[kind="secondary"] { background:#fff; border:1px solid var(--line); color:var(--navy); font-size:.82rem; }
.stButton>button[kind="secondary"]:hover { border-color:var(--aqua); color:var(--aqua); }

.st-key-resposta { background:#fff; border:1px solid var(--line); border-left:5px solid var(--aqua); border-radius:18px; padding:20px 24px; box-shadow:0 12px 32px rgba(11,47,91,.06); }
.st-key-resposta p, .st-key-resposta li { font-size:1.04rem; line-height:1.65; color:#183942; }
.st-key-resposta-recusa { background:#fff8e7; border:1px solid #f0d99c; border-left:5px solid #e0a82e; border-radius:18px; padding:20px 24px; }
.st-key-resposta-recusa p { font-size:1.04rem; line-height:1.6; color:#5c4710; }
.tag-recusa { display:inline-block; background:#fbe7b5; color:#7a5800; border-radius:999px; padding:3px 10px; font-size:.72rem; font-weight:800; margin-bottom:8px; }

.evidence-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:15px 17px; margin-bottom:11px; box-shadow:0 8px 22px rgba(11,47,91,.045); }
.source-line { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; margin-bottom:8px; }
.source-title { font-weight:700; color:var(--navy); font-size:.92rem; line-height:1.3; }
.source-meta { color:var(--muted); font-size:.76rem; font-weight:500; }
.page-pill { white-space:nowrap; background:var(--mist); color:var(--navy); border-radius:999px; padding:4px 10px; font-size:.72rem; font-weight:800; }
.quote { color:#526970; font-size:.88rem; line-height:1.5; border-left:2px solid #a9d9ec; padding-left:12px; }
.score { display:flex; align-items:center; gap:8px; color:#759097; font-size:.72rem; margin-top:9px; }
.bar { flex:1; max-width:140px; height:5px; background:#e6eff3; border-radius:99px; overflow:hidden; }
.bar span { display:block; height:100%; background:linear-gradient(90deg,#0a4f8c,#12a5dc); }
.empty { border:1.5px dashed #cfe0e8; border-radius:16px; padding:26px; color:var(--muted); text-align:center; font-size:.92rem; }

.footer-note { color:#6f878e; text-align:center; font-size:.8rem; line-height:1.5; margin-top:30px; padding-top:14px; border-top:1px solid var(--line); }
.footer-note b { color:var(--navy); }
@media (max-width:800px) { .hero img { height:70px; } .pipeline { grid-template-columns:1fr 1fr; } }
</style>
"""


def nome_modelo() -> str:
    modelo = get_generation_model()
    return NOMES_MODELOS.get(modelo, modelo)


def logo_html() -> str:
    dados = base64.b64encode(LOGO.read_bytes()).decode()
    return f'<img src="data:image/svg+xml;base64,{dados}" alt="IAqua-RAG">'


@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=200)
def consultar(pergunta: str):
    # Cache: perguntas repetidas (ex.: os exemplos) respondem na hora e sem custo.
    return responder(pergunta)


def usar_exemplo(texto: str) -> None:
    st.session_state.pergunta = texto
    st.session_state.executar = True


def render_evidencias(evidencias) -> None:
    for i, ev in enumerate(evidencias, start=1):
        meta = ", ".join(
            str(x)
            for x in (ev["instituicao"], ev["ano"])
            if x and str(x).strip().lower() not in ("não informado", "nao informado")
        )
        relevancia = max(0.0, min(1.0, float(ev["similaridade"])))
        st.markdown(
            f"""
            <div class="evidence-card">
              <div class="source-line">
                <div>
                  <div class="source-title">[{i}] {html.escape(str(ev["titulo"]))}</div>
                  <div class="source-meta">{html.escape(meta)}</div>
                </div>
                <div class="page-pill">p. {html.escape(str(ev["pagina"]))}</div>
              </div>
              <div class="quote">“{html.escape(ev["trecho"])}”</div>
              <div class="score">similaridade {relevancia:.2f}
                <div class="bar"><span style="width:{relevancia * 100:.0f}%"></span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="hero">
      {logo_html()}
      <div class="hero-text">
        <p class="subtitle">Consulta técnica em piscicultura de <b>tilápia</b> e
        <b>tambaqui</b> com respostas fundamentadas em documentos técnicos,
        citando documento e página.</p>
        <div class="badges">
          <span class="badge live">● DEMONSTRAÇÃO AO VIVO</span>
          <span class="badge">14 DOCUMENTOS TÉCNICOS</span>
          <span class="badge">EMBRAPA E MANUAIS</span>
          <span class="badge">RESPOSTAS COM CITAÇÃO</span>
        </div>
      </div>
    </div>
    <div class="pipeline">
      <div class="pipe-step"><b>1</b>Pergunta</div>
      <div class="pipe-step"><b>2</b>Busca nos documentos</div>
      <div class="pipe-step"><b>3</b>5 trechos mais relevantes</div>
      <div class="pipe-step"><b>4</b>Resposta com citação</div>
    </div>
    """,
    unsafe_allow_html=True,
)

esquerda, direita = st.columns([1.15, 1], gap="large")

with esquerda:
    st.markdown('<div class="eyebrow">Sua pergunta</div>', unsafe_allow_html=True)
    st.text_area(
        "Pergunta",
        key="pergunta",
        height=110,
        label_visibility="collapsed",
        placeholder="Ex.: Qual a temperatura ideal para criação de tilápia em tanques-rede?",
    )
    clicou = st.button("Consultar", type="primary", use_container_width=True)

    st.markdown('<div class="eyebrow" style="margin-top:14px">Exemplos</div>', unsafe_allow_html=True)
    colunas = st.columns(2) + st.columns(2)
    for coluna, (rotulo, texto) in zip(colunas, EXEMPLOS):
        coluna.button(
            rotulo,
            key=f"ex-{rotulo}",
            on_click=usar_exemplo,
            args=(texto,),
            use_container_width=True,
        )

    executar = clicou or st.session_state.pop("executar", False)
    if executar:
        pergunta_atual = (st.session_state.get("pergunta") or "").strip()
        faltando = [k for k in required_api_keys() if not os.environ.get(k)]
        if not pergunta_atual:
            st.warning("Digite uma pergunta.")
        elif faltando:
            st.error(
                "Erro de configuração: "
                + ", ".join(faltando)
                + " não definida(s) neste app. Configure em Settings → Secrets."
            )
        else:
            with st.spinner("Buscando nos documentos e redigindo a resposta..."):
                try:
                    st.session_state.resultado = consultar(pergunta_atual)
                except Exception as e:
                    st.session_state.resultado = None
                    st.error(f"Erro ao consultar o sistema: {e}")

    resultado = st.session_state.get("resultado")
    if resultado:
        st.markdown('<div class="eyebrow" style="margin-top:20px">Resposta</div>', unsafe_allow_html=True)
        resposta = resultado["resposta"]
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

with direita:
    st.markdown('<div class="eyebrow">Evidências recuperadas</div>', unsafe_allow_html=True)
    resultado = st.session_state.get("resultado")
    if resultado:
        render_evidencias(resultado["evidencias"])
    else:
        st.markdown(
            '<div class="empty">Os trechos dos documentos usados na resposta '
            "aparecem aqui, com documento, página e similaridade.</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    f"""
    <div class="footer-note">
      Demonstração ao vivo: geração das respostas com <b>{html.escape(nome_modelo())}</b>.
      Os resultados do estudo apresentados no banner foram obtidos com
      <b>{MODELO_DO_ESTUDO}</b>.<br>
      Protótipo de pesquisa — confira sempre a fonte citada antes de usar a resposta.
    </div>
    """,
    unsafe_allow_html=True,
)
