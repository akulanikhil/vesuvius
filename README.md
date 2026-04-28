# Vesuvius Discord Study

Analysis of the Vesuvius Challenge Discord server to uncover discussion themes, collaboration patterns, and sentiment trends using topic modeling and Retrieval-Augmented Generation (RAG).

---

## Repository Structure

```
vesuvius_discord_study/
├── JSON_filter/       # Filter and clean raw Discord JSON exports by date range
├── LDA/               # Early LDA topic modeling and pyLDAvis visualization
├── term_level/        # Term-frequency analysis and bigram LDA topic modeling (main analysis)
├── RAG/               # Vector embedding, ChromaDB storage, and LLM-based Q&A
├── LLaMA/             # LLaMA inference scripts
└── TXT/               # JSON-to-plaintext conversion utilities
```

---

## Modules

**`JSON_filter/`** — Filters raw exports to a target date range (Mar–Oct 2023), normalizes timestamps, and removes empty files. Run `filter.py` then `move_empty.py`.

**`LDA/`** — Early exploration using Gensim LDA on a single channel. Outputs per-message topic assignments and an interactive pyLDAvis dashboard via `visualize_metadata.py`.

**`term_level/`** — Main analysis. Contains three notebooks for term-frequency and topic modeling — see [`term_level/README.md`](term_level/README.md) for full details.

**`RAG/`** — Groups messages into time-proximity chunks, embeds them into a Chroma vector store, and runs a LangGraph Q&A pipeline using a local Ollama LLM.

---

## Dependencies

```bash
pip install spacy nltk pandas scikit-learn gensim pyLDAvis \
            matplotlib seaborn streamlit langchain langgraph \
            langchain-chroma langchain-huggingface openpyxl
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```
