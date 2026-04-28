# Term-Level Analysis & Topic Modeling

This folder contains three notebooks that together cover term-frequency analysis and LDA topic modeling on the Vesuvius Challenge Discord data (`#general` and `#papyrology` channels).

---

## Notebooks

| Notebook | Author | Purpose |
|---|---|---|
| `term_level_streamlined.ipynb` | Nikhil | Term-frequency analysis (TF/DF + MI) |
| `term_level_revised.ipynb` | Summer | Preprocessing, term frequency, and statistics |
| `discord_bigram_lda.ipynb` | Both | Bigram LDA topic modeling (k=10–30) |

---

## 1. `term_level_streamlined.ipynb` — Nikhil's Term-Frequency Analysis

### What it does
Runs a full term-level analysis pipeline on the filtered Discord JSON files:
1. **Preprocessing** — lowercases, lemmatizes (spaCy), strips URLs/emojis/file extensions/numbers, removes punctuation
2. **Stop word removal** — combines NLTK English stopwords + `stopwords-iso.json` + a custom list of names and domain noise words
3. **Greek detection** — any token containing a Unicode Greek character is replaced with `[GREEK_LANGUAGE]`
4. **Tokenization** — produces per-document token lists (alpha-only, min length 2)
5. **Bigram generation** — sliding window bigrams per document
6. **Counting** — unigram and bigram TF (term frequency) and DF (document frequency)
7. **Mutual Information** — PMI scores for bigrams to surface the most meaningful collocations
8. **Export** — saves results to `outputs/` as CSV and `.xlsx`

### Configuration
Edit the `Config` class at the top of the notebook to adjust behaviour:

```python
class Config:
    lowercase        = True
    keep_only_alpha  = True
    min_token_len    = 2
    use_lemma        = True        # requires spaCy en_core_web_sm
    remove_numbers   = True
    use_nltk_stopwords = True
    extra_stopwords  = { "server", "joined", "scroll", ... }
```

To add or remove custom stop words, edit `extra_stopwords` in the `Config` class.

### How to run

**Install dependencies (first time only):**
```bash
pip install spacy nltk pandas tqdm openpyxl xlsxwriter
python -m spacy download en_core_web_sm
```

**Set the input path** — in Cell 11, update `folder` to point to your filtered JSON directory:
```python
folder = "./filtered_JSON"   # or filtered_JSON_all for all channels
```

**Run all cells top to bottom.** Outputs are written to:
```
outputs/
  unigrams.csv                     # unigram TF + DF
  bigrams.csv                      # bigram TF + DF
  bigrams_mi.csv                   # bigrams ranked by Mutual Information
  bigrams_single_corpus.csv        # bigrams across combined corpus
  mi_bigrams_general_vs_papyrology.csv
  term_level_analysis.xlsx         # all of the above + raw messages sheet
```

---

## 2. `term_level_revised.ipynb` — Summer's Term-Frequency & Statistics

### What it does
A two-phase notebook: preprocessing then analysis.

**Phase 1 — Preprocessing (Cells 0–2)**
1. Loads `stopwords-iso.json` and merges with NLTK English stopwords and a custom list
2. Lemmatizes and lowercases each message with spaCy
3. Strips URLs, numbers, punctuation, and emojis
4. Saves preprocessed messages to `preprocessedJSON/` as JSON files
5. Cell 2 provides an **additive stop word patch** — add new words to `additional_stopwords` and re-run just that cell without reprocessing everything from scratch

**Phase 2 — Term Frequency & Statistics (Cells 3–8)**
1. Loads preprocessed JSON from `preprocessedJSON/`, filtered to `#general` and `#papyrology` only
2. Detects Greek script tokens (via `unicodedata`) and replaces them with `[GREEK_LANGUAGE]`
3. Computes unigram and bigram TF + DF using NLTK's `ngrams`
4. Cell 8 reads `channel_analysis.xlsx` and produces summary statistics and plots broken down by channel (general, papyrology, both)

### How to run

**Install dependencies (first time only):**
```bash
pip install spacy nltk pandas matplotlib seaborn openpyxl
python -m spacy download en_core_web_sm
```

**Input files needed before running:**
- `stopwords-iso.json` — must be in the same directory as the notebook
- Filtered JSON files in `./filtered_JSON/` (or update the path in Cell 1)

**Run Cells 0–2 first** to generate `preprocessedJSON/`. This only needs to be done once unless the source data changes.

**To add stop words later** — edit `additional_stopwords` in Cell 2 and re-run only that cell.

**Run Cells 3–8** for term frequency and statistics. Cell 8 requires `channel_analysis.xlsx` to already exist in the working directory.

**Key path to update** — Cell 1 writes preprocessed output to `preprocessedJSON/`. If you move the notebook, update `input_folder` and `output_folder` accordingly.

---

## 3. `discord_bigram_lda.ipynb` — Bigram LDA Topic Modeling

### What it does
Runs a full LDA topic modeling pipeline on the `#general` and `#papyrology` channels:

1. **Loading** — reads filtered JSON files, extracts `Default`-type messages with timestamps and channel info
2. **Cleaning** — removes URL-only messages, system messages (join notices, thread starts), and messages under 10 characters
3. **Preprocessing** — same pipeline as `term_level_streamlined`: lowercase, lemmatize, strip URLs/emojis/numbers, custom stop words, Greek token replacement
4. **Time-window aggregation** — groups messages into **30-minute buckets per channel**, treating each window as one LDA document (produces ~3,500 documents from ~1,500 raw windows per channel)
5. **Vectorization** — `CountVectorizer` with a custom analyzer that runs the preprocessing pipeline, producing a bigram matrix (vocabulary: ~6,158 bigrams, `min_df=2`, `max_df=0.90`)
6. **LDA fitting** — runs `LatentDirichletAllocation` for k = 10, 20, 25, 30 topics
7. **Evaluation** — computes UMass coherence scores per topic
8. **Export** — saves one CSV per k value to `outputs_rq2/`

### How to run

**Install dependencies (first time only):**
```bash
pip install spacy nltk pandas scikit-learn
python -m spacy download en_core_web_sm
```

**Set the input directory** — in Cell 7, update `input_dir`:
```python
outputs = run_pipeline(
    input_dir="./filtered_JSON",   # folder containing general + papyrology JSONs
    output_dir="./outputs_rq2",
    aggregate_window="30min",      # change to "60min" to test wider windows
    ks=(10, 20, 25, 30),
    ...
)
```

**Run all cells top to bottom.** The pipeline will print progress at each step:
```
[1/5] Load & clean…
[2/5] Aggregating by 30min per channel…
Documents for LDA: 3,536
[3/5] Building bigram matrix…
Vocabulary size (bigrams): 6,158
[4/5] Fitting LDA with k=20…
[4/5] Fitting LDA with k=25…
...
[5/5] Done ✓
```

**Outputs written to `outputs_rq2/`:**
```
topics_k10.csv
topics_k20.csv
topics_k25.csv
topics_k30.csv
```

Each CSV has one row per topic with:

| Column | Description |
|---|---|
| `topic` | Topic ID (0 to k-1) |
| `docs_hard_count` | # windows assigned to this topic (argmax) |
| `docs_mean_weight` | Mean topic weight across all documents |
| `coherence_umass` | UMass coherence score (higher = more coherent) |
| `top_terms` | Top 15 bigrams |
| `top_terms_with_weights` | Top 15 bigrams with LDA weights |
| `docs_papyrology` | # windows from `#papyrology` |
| `docs_general` | # windows from `#general` |
| `sample_messages` | 3 representative message snippets |

### Tuning
- **Change k** — add or remove values from `ks=(10, 20, 25, 30)`
- **Change window size** — set `aggregate_window="60min"` for wider context windows
- **Switch to TF-IDF** — set `use_tfidf=True` (results saved to `outputs_rq2_tfidf/`)
- **Add stop words** — edit `BIGRAM_STOPWORDS` in Cell 4

---

## Shared Dependencies

All three notebooks require the same core stack:

```bash
pip install spacy nltk pandas scikit-learn matplotlib seaborn openpyxl xlsxwriter
python -m spacy download en_core_web_sm
```

And the same `stopwords-iso.json` file, which must be present in the `term_level/` directory.

---

## Folder Structure

```
term_level/
├── term_level_streamlined.ipynb   # Nikhil — TF/DF/MI analysis
├── term_level_revised.ipynb       # Summer — preprocessing + stats
├── discord_bigram_lda.ipynb       # LDA topic modeling
├── stopwords-iso.json             # ISO stop word list (required by all notebooks)
├── filtered_JSON/                 # general + papyrology filtered JSONs
├── filtered_JSON_all/             # all channel filtered JSONs
├── preprocessedJSON/              # output of term_level_revised preprocessing step
├── outputs/                       # term_level_streamlined outputs
│   ├── unigrams.csv
│   ├── bigrams.csv
│   ├── bigrams_mi.csv
│   └── term_level_analysis.xlsx
├── outputs_rq2/                   # LDA outputs (raw counts)
│   ├── topics_k10.csv
│   ├── topics_k20.csv
│   ├── topics_k25.csv
│   └── topics_k30.csv
├── outputs_rq2_tfidf/             # LDA outputs (TF-IDF weighted)
└── figures/                       # visualizations (from visualizations.py)
    ├── fig2_coherence_vs_docs.png
    ├── fig4_channel_split.png
    └── fig7_topic_activity_over_time.png
```
