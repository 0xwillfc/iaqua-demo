"""Interface web do IAqua para demonstracao ao vivo (Streamlit Community Cloud).

Roda sobre o mesmo src/rag.py do projeto original, apontando para um
data/chroma_db ja indexado (nao requer os 14 PDFs nem reingestao).
"""
import os

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

from src.rag import responder  # noqa: E402

EXEMPLOS = [
    "Quais variáveis de qualidade da água devem ser monitoradas diariamente na "
    "produção de tilápia em tanques-rede?",
    "Quais parâmetros de qualidade da água são classificados como padrões CONAMA "
    "classe 2 relevantes para a piscicultura e qual deles também é monitorado "
    "diariamente nas boas práticas de manejo?",
    "Como as tilápias regulam a própria temperatura corporal para manter o "
    "consumo de ração estável ao longo do ano?",
    "Qual é o preço atual do quilo do filé de tilápia no mercado brasileiro?",
]

st.set_page_config(page_title="IAqua", page_icon="🐟")

st.title("🐟 IAqua")
st.markdown(
    "Sistema de recuperação aumentada por geração (RAG) para consulta técnica "
    "na criação de **tilápia** e **tambaqui**. As respostas são fundamentadas "
    "em documentos técnicos (Embrapa e manuais), com citação de documento e "
    "página.\n\n"
    "_Demonstração — confira sempre a fonte citada antes de usar a resposta._"
)

with st.expander("Exemplos de pergunta (clique para usar)"):
    for ex in EXEMPLOS:
        if st.button(ex, key=ex):
            st.session_state["pergunta"] = ex

pergunta = st.text_area(
    "Pergunta",
    key="pergunta",
    placeholder=(
        "Ex.: Qual a temperatura ideal para criação de tilápia em tanques-rede?"
    ),
)

if st.button("Consultar", type="primary"):
    pergunta_atual = (pergunta or "").strip()
    if not pergunta_atual:
        st.warning("Digite uma pergunta.")
    elif not os.environ.get("NVIDIA_API_KEY"):
        st.error(
            "Erro de configuração: a variável NVIDIA_API_KEY não está definida "
            "neste app. Configure-a em Settings → Secrets."
        )
    else:
        with st.spinner("Consultando..."):
            try:
                resultado = responder(pergunta_atual)
            except Exception as e:
                st.error(f"Erro ao consultar o sistema: {e}")
                resultado = None

        if resultado:
            st.markdown("### Resposta")
            st.markdown(resultado["resposta"])

            st.markdown("### Evidências recuperadas")
            for i, ev in enumerate(resultado["evidencias"], start=1):
                st.markdown(
                    f"**[{i}] {ev['titulo']}** ({ev['instituicao']}, {ev['ano']}) — "
                    f"p. {ev['pagina']} · similaridade {ev['similaridade']}"
                )
                st.caption(ev["trecho"])
