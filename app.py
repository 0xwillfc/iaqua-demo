"""Interface web do IAqua para demonstracao ao vivo (Hugging Face Spaces).

Roda sobre o mesmo src/rag.py do projeto original, apontando para um
data/chroma_db ja indexado (nao requer os 14 PDFs nem reingestao).
"""
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

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


def consultar(pergunta: str):
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "Digite uma pergunta.", ""
    if not os.environ.get("NVIDIA_API_KEY"):
        return (
            "Erro de configuração: a variável NVIDIA_API_KEY não está definida "
            "neste Space. Configure-a em Settings → Variables and secrets.",
            "",
        )
    try:
        resultado = responder(pergunta)
    except Exception as e:  # nunca deixar a demo travar em tela de erro cru
        return f"Erro ao consultar o sistema: {e}", ""

    linhas_evidencia = ["### Evidências recuperadas\n"]
    for i, ev in enumerate(resultado["evidencias"], start=1):
        linhas_evidencia.append(
            f"**[{i}] {ev['titulo']}** ({ev['instituicao']}, {ev['ano']}) — "
            f"p. {ev['pagina']} · similaridade {ev['similaridade']}\n\n"
            f"> {ev['trecho']}\n"
        )
    return resultado["resposta"], "\n".join(linhas_evidencia)


with gr.Blocks(title="IAqua") as demo:
    gr.Markdown(
        "# 🐟 IAqua\n"
        "Sistema de recuperação aumentada por geração (RAG) para consulta técnica "
        "na criação de **tilápia** e **tambaqui**. As respostas são fundamentadas "
        "em documentos técnicos (Embrapa e manuais), com citação de documento e "
        "página.\n\n"
        "_Demonstração — confira sempre a fonte citada antes de usar a resposta._"
    )
    caixa_pergunta = gr.Textbox(
        label="Pergunta",
        placeholder=(
            "Ex.: Qual a temperatura ideal para criação de tilápia em "
            "tanques-rede?"
        ),
        lines=2,
    )
    botao = gr.Button("Consultar", variant="primary")
    saida_resposta = gr.Markdown(label="Resposta")
    saida_evidencias = gr.Markdown(label="Evidências")

    gr.Examples(examples=EXEMPLOS, inputs=caixa_pergunta)

    botao.click(
        consultar, inputs=caixa_pergunta, outputs=[saida_resposta, saida_evidencias]
    )
    caixa_pergunta.submit(
        consultar, inputs=caixa_pergunta, outputs=[saida_resposta, saida_evidencias]
    )


if __name__ == "__main__":
    demo.launch()
