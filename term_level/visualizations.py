"""
visualizations.py
-----------------
Produces three figures for the Topic Modeling section of the research paper:

  Chart 2 — Scatter: coherence score vs. document count (colored by category)
  Chart 4 — Stacked bar: #general vs. #papyrology doc split per topic
  Chart 7 — Line: topic-category activity over time (re-runs LDA k=30)

Run:
    python3 visualizations.py

Outputs (saved to ./figures/):
    fig2_coherence_vs_docs.png
    fig4_channel_split.png
    fig7_topic_activity_over_time.png

Requirements:
    pip install matplotlib seaborn pandas scikit-learn spacy
    python -m spacy download en_core_web_sm
"""

import os
import re
import json
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
from typing import List, Set

warnings.filterwarnings("ignore")

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV_K30   = "./outputs_rq2/topics_k30.csv"
JSON_DIR  = "./filtered_JSON"   # general + papyrology JSONs

# ── Topic metadata ────────────────────────────────────────────────────────────
TOPIC_LABELS = {
    0:  "General / Catch-all",
    1:  "Segmentation Team Coordination",
    2:  "Ink Detection & Papyrology",
    3:  "Ink Analysis & Segmentation",
    4:  "Greek Transcription",
    5:  "VC Software & Versioning",
    6:  "Competition & CT Data",
    7:  "Ink Detection Methods",
    8:  "Team Resource Sharing",
    9:  "Segmentation Workflow",
    10: "Celebrating Breakthroughs",
    11: "Ancient Greek & Surface Normals",
    12: "Framework & Problem-Solving",
    13: "Data Quality Discussion",
    14: "Segmentation Algorithm Refinement",
    15: "ML Ink Detection",
    16: "Pen Tool & Blog Updates",
    17: "Milestone Announcements",
    18: "Monster Segments",
    19: "VC Feature Requests",
    20: "Collaborative Idea Sharing",
    21: "Contract Segmenters",
    22: "Docker Environment Setup",
    23: "Segment Rendering & Coordination",
    24: "Model Training & Flattening",
    25: "Community Excitement / Experiments",
    26: "Crack Texture & Auto-Segmentation",
    27: "Ink Labels & Prize Discussion",
    28: "Onboarding & Collaboration",
    29: "Named Contributor Recognition",
}

TOPIC_CATEGORIES = {
    0:  "Catch-all",
    1:  "Collaboration",
    2:  "Technical",
    3:  "Technical",
    4:  "Technical",
    5:  "Technical",
    6:  "Technical",
    7:  "Technical",
    8:  "Collaboration",
    9:  "Technical",
    10: "Encouragement",
    11: "Technical",
    12: "Collaboration",
    13: "Technical",
    14: "Technical",
    15: "Technical",
    16: "Technical",
    17: "Encouragement",
    18: "Technical",
    19: "Technical",
    20: "Collaboration",
    21: "Technical",
    22: "Technical",
    23: "Technical",
    24: "Technical",
    25: "Encouragement",
    26: "Technical",
    27: "Technical",
    28: "Collaboration",
    29: "Encouragement",
}

CATEGORY_COLORS = {
    "Technical":     "#4C72B0",
    "Collaboration": "#55A868",
    "Encouragement": "#C44E52",
    "Catch-all":     "#AAAAAA",
}

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Scatter: coherence vs. document count
# ─────────────────────────────────────────────────────────────────────────────
def chart2_coherence_vs_docs():
    print("Generating Chart 2: Coherence vs. Document Count...")
    df = pd.read_csv(CSV_K30)
    df["category"] = df["topic"].map(TOPIC_CATEGORIES)
    df["label"]    = df["topic"].map(TOPIC_LABELS)
    df["color"]    = df["category"].map(CATEGORY_COLORS)

    # Exclude Topic 0 (catch-all outlier) so the rest of the distribution is visible
    outlier = df[df["topic"] == 0].iloc[0]
    df_plot = df[df["topic"] != 0].copy()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for cat, grp in df_plot.groupby("category"):
        ax.scatter(
            grp["docs_hard_count"],
            grp["coherence_umass"],
            c=CATEGORY_COLORS[cat],
            label=cat,
            s=100,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

    # Annotate all collaboration and encouragement topics, plus a few notable technical ones
    highlight = {10, 12, 17, 20, 25, 28, 1, 8, 22, 4}
    # Nudge map: topic -> (dx, dy) in points to avoid overlap
    nudges = {
        10: (8,   4),
        12: (8,   4),
        17: (8,  -12),
        20: (8,   4),
        25: (8,  -12),
        28: (8,   4),
        1:  (-130, -12),
        8:  (8,   4),
        22: (8,   4),
        4:  (8,  -12),
    }
    for _, row in df_plot.iterrows():
        t = int(row["topic"])
        if t in highlight:
            dx, dy = nudges.get(t, (8, 4))
            ax.annotate(
                f"T{t}: {TOPIC_LABELS[t]}",
                xy=(row["docs_hard_count"], row["coherence_umass"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                color="#222222",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7) if abs(dx) > 20 else None,
            )

    # Mean coherence line (computed on the non-outlier set for relevance)
    mean_coh = df_plot["coherence_umass"].mean()
    ax.axhline(mean_coh, color="gray", linestyle="--", linewidth=0.9,
               label=f"Mean coherence ({mean_coh:.2f})")

    # Footnote about excluded outlier
    ax.annotate(
        f"* Topic 0 (General / Catch-all) excluded — {int(outlier['docs_hard_count']):,} docs, "
        f"coherence {outlier['coherence_umass']:.2f}",
        xy=(0.01, 0.02), xycoords="axes fraction",
        fontsize=8, color="#666666",
    )

    ax.set_xlabel("Document Count (# 30-min windows assigned to topic)", fontsize=11)
    ax.set_ylabel("UMass Coherence Score", fontsize=11)
    ax.set_title("Topic Coherence vs. Document Count — k=30 (Topic 0 excluded)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    legend_patches = [
        mpatches.Patch(color=c, label=cat)
        for cat, c in CATEGORY_COLORS.items()
        if cat != "Catch-all"
    ]
    legend_patches.append(
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Mean coherence")
    )
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_coherence_vs_docs.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Stacked bar: #general vs. #papyrology split per topic
# ─────────────────────────────────────────────────────────────────────────────
def chart4_channel_split():
    print("Generating Chart 4: Channel Split per Topic...")
    df = pd.read_csv(CSV_K30)
    df["category"] = df["topic"].map(TOPIC_CATEGORIES)
    df["label"]    = df["topic"].map(TOPIC_LABELS)

    # Exclude Topic 0 catch-all
    t0 = df[df["topic"] == 0].iloc[0]
    df_plot = df[df["topic"] != 0].copy()

    # Compute papyrology share and sort by it descending
    df_plot["total"]      = df_plot["docs_general"] + df_plot["docs_papyrology"]
    df_plot["papy_share"] = df_plot["docs_papyrology"] / df_plot["total"].replace(0, np.nan)
    df_plot = df_plot.sort_values("papy_share", ascending=False).reset_index(drop=True)

    # Short readable x-labels: "T4: Greek Transcription"
    x_labels   = [f"T{int(r.topic)}: {TOPIC_LABELS[int(r.topic)]}" for _, r in df_plot.iterrows()]
    general_vals = df_plot["docs_general"].values
    papyro_vals  = df_plot["docs_papyrology"].values
    categories   = df_plot["category"].values

    x = np.arange(len(df_plot))
    bar_width = 0.7

    fig, ax = plt.subplots(figsize=(20, 7))

    ax.bar(x, papyro_vals,  bar_width, label="#papyrology", color="#DD8452", alpha=0.85)
    ax.bar(x, general_vals, bar_width, label="#general",    color="#4C72B0", alpha=0.85,
           bottom=papyro_vals)

    # x-tick labels: full topic name, colored and bolded by category
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8.5, rotation=40, ha="right")
    for tick, cat in zip(ax.get_xticklabels(), categories):
        tick.set_color(CATEGORY_COLORS[cat])
        tick.set_fontweight("bold" if cat in ("Collaboration", "Encouragement") else "normal")

    # Add papyrology % label on top of each bar
    for i, (gen, pap) in enumerate(zip(general_vals, papyro_vals)):
        total = gen + pap
        if total > 0 and pap > 0:
            pct = pap / total * 100
            ax.text(i, total + 1.5, f"{pct:.0f}%",
                    ha="center", va="bottom", fontsize=7, color="#CC5500")

    # Single clean legend: channels + category color key side by side
    channel_patches = [
        mpatches.Patch(color="#4C72B0", alpha=0.85, label="#general"),
        mpatches.Patch(color="#DD8452", alpha=0.85, label="#papyrology"),
    ]
    cat_patches = [
        mpatches.Patch(color=CATEGORY_COLORS["Technical"],     label="Technical (label color)"),
        mpatches.Patch(color=CATEGORY_COLORS["Collaboration"], label="Collaboration (label color)"),
        mpatches.Patch(color=CATEGORY_COLORS["Encouragement"], label="Encouragement (label color)"),
    ]
    leg1 = ax.legend(handles=channel_patches, loc="upper right",
                     fontsize=9, title="Bar fill", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=cat_patches, loc="upper center",
              fontsize=9, title="X-label color key", framealpha=0.9, ncol=3)

    ax.set_xlabel("Topic (sorted by #papyrology share, high → low)", fontsize=11)
    ax.set_ylabel("Document Count (30-min windows)", fontsize=11)
    ax.set_title(
        "#general vs. #papyrology Distribution per Topic — k=30, sorted by papyrology share\n"
        "Orange % = proportion of docs from #papyrology channel",
        fontsize=12, fontweight="bold"
    )

    t0_note = (f"* Topic 0 (General / Catch-all) excluded — "
               f"{int(t0['docs_general']):,} general / {int(t0['docs_papyrology']):,} papyrology docs")
    ax.annotate(t0_note, xy=(0.01, 0.98), xycoords="axes fraction",
                fontsize=8, color="#666666", va="top")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_channel_split.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 7 — Topic category activity over time (re-runs LDA k=30)
# ─────────────────────────────────────────────────────────────────────────────

# --- Preprocessing (mirrors discord_bigram_lda.ipynb) ---
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY = True
except Exception:
    _SPACY = False
    print("  [warn] spaCy not available — lemmatization skipped for chart 7")

try:
    from nltk.corpus import stopwords as _sw
    _STOP_EN: Set[str] = set(_sw.words("english"))
except Exception:
    _STOP_EN: Set[str] = set()

_CUSTOM_STOP: Set[str] = {
    "server", "joined", "scroll", "papyrus", "image",
    "brett", "olsen", "entry", "start", "thread",
    "moshe", "levy", "casey", "handmer", "mae",
    "sawatzky", "hari", "seldon", "ben",
}
_ALL_STOPS: Set[str] = _STOP_EN | _CUSTOM_STOP

_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION  = re.compile(r"@\S+")

def _contains_greek(tok: str) -> bool:
    return any("GREEK" in unicodedata.name(c, "") for c in tok)

def _preprocess(text: str) -> str:
    """Return a cleaned string of space-separated tokens (bigrams built by vectorizer)."""
    text = _URL_RE.sub("", text)
    text = _MENTION.sub("", text)
    tokens: List[str] = []
    raw_toks = text.split()
    for tok in raw_toks:
        tok = tok.strip(".,!?;:\"'()[]{}|")
        if not tok:
            continue
        if _contains_greek(tok):
            tokens.append("GREEK_LANGUAGE")
            continue
        if not tok.isalpha():
            continue
        tok_lower = tok.lower()
        if tok_lower in _ALL_STOPS or len(tok_lower) < 2:
            continue
        if _SPACY:
            doc = _nlp(tok_lower)
            lemma = doc[0].lemma_ if doc else tok_lower
            if lemma in _ALL_STOPS or len(lemma) < 2:
                continue
            tokens.append(lemma)
        else:
            tokens.append(tok_lower)
    return " ".join(tokens)

def _bigram_analyzer(text: str) -> List[str]:
    toks = text.split()
    if len(toks) < 2:
        return toks
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]

def _load_json_dir(json_dir: str) -> pd.DataFrame:
    rows = []
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        channel_name = fname.split(" - ")[-1].split(" [")[0].strip()
        with open(os.path.join(json_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        for msg in data.get("messages", []):
            if msg.get("type") != "Default":
                continue
            content = msg.get("content", "").strip()
            if not content:
                continue
            rows.append({
                "timestamp":    msg.get("timestamp"),
                "channel_name": channel_name,
                "text":         content,
            })
    return pd.DataFrame(rows)

def chart7_activity_over_time():
    print("Generating Chart 7: Topic Category Activity Over Time...")
    print("  Loading and preprocessing messages (this may take a minute)...")

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    df = _load_json_dir(JSON_DIR)
    if df.empty:
        print(f"  [error] No messages found in {JSON_DIR}")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["bucket_start"] = df["timestamp"].dt.floor("30min")
    df["clean"] = df["text"].map(_preprocess)

    # Aggregate into 30-min windows per channel
    df_docs = (
        df.sort_values("timestamp")
          .groupby(["channel_name", "bucket_start"], as_index=False)
          .agg(text=("clean", lambda s: " ".join(s)), n_msgs=("text", "count"))
    )
    df_docs = df_docs[df_docs["text"].str.len() >= 10].reset_index(drop=True)

    print(f"  Documents: {len(df_docs):,}  |  Fitting LDA k=30...")
    vect = CountVectorizer(analyzer=_bigram_analyzer, lowercase=False, min_df=2, max_df=0.90)
    X    = vect.fit_transform(df_docs["text"])

    lda = LatentDirichletAllocation(
        n_components=30, learning_method="batch",
        random_state=42, max_iter=50
    )
    lda.fit(X)
    theta = lda.transform(X)

    df_docs["dominant_topic"] = theta.argmax(axis=1)
    df_docs["category"]       = df_docs["dominant_topic"].map(TOPIC_CATEGORIES).fillna("Technical")
    df_docs["week"]           = df_docs["bucket_start"].dt.to_period("W").dt.start_time

    # Weekly document counts per category
    weekly = (
        df_docs.groupby(["week", "category"])
               .size()
               .reset_index(name="doc_count")
    )
    pivot = weekly.pivot(index="week", columns="category", values="doc_count").fillna(0)

    # Ensure all categories present
    for cat in CATEGORY_COLORS:
        if cat not in pivot.columns:
            pivot[cat] = 0
    pivot = pivot[["Technical", "Collaboration", "Encouragement", "Catch-all"]]

    fig, ax = plt.subplots(figsize=(14, 6))

    for cat in ["Technical", "Collaboration", "Encouragement", "Catch-all"]:
        lw = 2.5 if cat in ("Collaboration", "Encouragement") else 1.5
        ls = "-"
        ax.plot(
            pivot.index, pivot[cat],
            color=CATEGORY_COLORS[cat],
            linewidth=lw,
            linestyle=ls,
            label=cat,
            alpha=0.9,
        )

    # Shade Collaboration + Encouragement for visual emphasis
    ax.fill_between(pivot.index, pivot["Collaboration"],
                    alpha=0.12, color=CATEGORY_COLORS["Collaboration"])
    ax.fill_between(pivot.index, pivot["Encouragement"],
                    alpha=0.12, color=CATEGORY_COLORS["Encouragement"])

    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("30-min Window Count", fontsize=11)
    ax.set_title("Topic Category Activity Over Time (k=30, weekly aggregation)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Rotate x-axis dates
    fig.autofmt_xdate(rotation=35)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig7_topic_activity_over_time.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chart2_coherence_vs_docs()
    chart4_channel_split()
    chart7_activity_over_time()
    print("\nAll figures saved to ./figures/")
