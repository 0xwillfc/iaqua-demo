# IAqua — demonstração

Interface web mínima para demonstrar o sistema RAG de consulta técnica em
piscicultura de tilápia e tambaqui. Usa o índice vetorial (ChromaDB) já
construído a partir dos 14 documentos técnicos — não reingesta nada.

Duas interfaces prontas, mesmo backend (`src/rag.py`):

- **`app_streamlit.py`** — recomendada. Deploy gratuito e persistente via
  [Streamlit Community Cloud](https://share.streamlit.io) (verificado
  gratuito em agosto de 2026: 1 GB RAM, dorme só após 12h sem uso).
- **`app.py`** (Gradio) — alternativa para rodar em Google Colab com
  `demo.launch(share=True)`, caso prefira não usar GitHub/Streamlit Cloud.

## ⚠️ Aviso importante sobre os modelos

Os modelos originais do estudo (`nv-embedqa-e5-v5` e `llama-3.1-70b-instruct`)
**foram desativados pela NVIDIA em agosto de 2026** (erro 410 "end of life").
Esta demo já usa substitutos testados e funcionais com a mesma chave, e o
índice em `data/chroma_db` já foi reconstruído com o novo modelo de embedding.
Se algum desses também for desativado no futuro, rode `client.models.list()`
com sua chave para achar um substituto disponível na conta antes de reindexar
(embeddings de modelos diferentes não são comparáveis — reindexação completa
é obrigatória a cada troca).

## Deploy no Streamlit Community Cloud

1. Suba esta pasta para um repositório no **GitHub** (veja comandos abaixo).
2. Em [share.streamlit.io](https://share.streamlit.io), conecte o repositório
   e escolha `app_streamlit.py` como arquivo principal.
3. Em **Settings → Secrets** do app, cole (formato TOML, não é tabela):

   ```toml
   NVIDIA_API_KEY = "sua_chave_aqui"
   EMBEDDING_PROVIDER = "nvidia"
   GENERATION_PROVIDER = "nvidia"
   NVIDIA_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
   NVIDIA_GENERATION_MODEL = "meta/llama-3.2-11b-vision-instruct"
   ```

Nunca coloque a chave diretamente no código, no requirements.txt, ou em
`.streamlit/secrets.toml` versionado (esse arquivo já está no `.gitignore`).

## Comandos git (repositório já iniciado localmente)

```bash
git remote add origin https://github.com/SEU-USUARIO/iaqua-demo.git
git push -u origin master
```
