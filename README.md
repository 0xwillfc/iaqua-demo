---
title: IAqua
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# IAqua — demonstração

Interface web mínima para demonstrar o sistema RAG de consulta técnica em
piscicultura de tilápia e tambaqui. Usa o índice vetorial (ChromaDB) já
construído a partir dos 14 documentos técnicos — não reingesta nada.

## Configuração necessária

⚠️ **Os modelos originais do estudo (`nv-embedqa-e5-v5` e `llama-3.1-70b-instruct`)
foram desativados pela NVIDIA em agosto de 2026** (erro 410 "end of life"). Esta
demo usa substitutos testados e funcionais com a mesma chave: o índice vetorial
em `data/chroma_db` já foi reconstruído com o novo modelo de embedding — não
reingesta nada, só configure as variáveis abaixo.

Em **Settings → Variables and secrets**, adicione como *secret*:

| Nome | Valor |
|---|---|
| `NVIDIA_API_KEY` | a mesma chave do projeto original |
| `EMBEDDING_PROVIDER` | `nvidia` |
| `GENERATION_PROVIDER` | `nvidia` |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nemotron-3-embed-1b` |
| `NVIDIA_GENERATION_MODEL` | `meta/llama-3.2-11b-vision-instruct` |

As duas últimas são obrigatórias — os valores padrão no `config.py` ainda
apontam para os modelos desativados. Nunca coloque a chave diretamente no
código ou no requirements.txt.

Se algum desses modelos também for desativado no futuro, rode
`client.models.list()` (ver `docs` da conversa) para achar um substituto
disponível na conta antes de reindexar.
