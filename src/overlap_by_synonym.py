import os
import json
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import ENTITY_TYPES
from src.utils import ensure_dir

# Répertoires
DEBUG_FILE = Path("outputs/debug_by_synonym.json")
OUT_DIR = Path("outputs/overlap_analysis/synonym_level")
ensure_dir(OUT_DIR)

def extract_spans(debug_data, code, synonym):
    """Récupère les ensembles de spans prédits pour un synonyme donné."""
    spans_by_text = {}
    key = f"{code}__{synonym}"
    for entry in debug_data.get(key, []):
        tid = entry["text_id"]
        span = tuple(entry["span"])
        spans_by_text.setdefault(tid, set()).add(span)
    return spans_by_text

def flatten_spans(spans_dict):
    """Concatène tous les spans en un seul set."""
    all_spans = set()
    for spans in spans_dict.values():
        all_spans.update(spans)
    return all_spans

def compute_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def main():
    with open(DEBUG_FILE, encoding="utf-8") as f:
        debug_data = json.load(f)

    for code, synonyms in ENTITY_TYPES.items():
        print(f" Overlap des prédictions pour : {code}")
        span_sets = {
            syn: flatten_spans(extract_spans(debug_data, code, syn))
            for syn in synonyms
        }

        matrix = np.zeros((len(synonyms), len(synonyms)))
        for i, j in itertools.product(range(len(synonyms)), repeat=2):
            s1, s2 = synonyms[i], synonyms[j]
            matrix[i, j] = compute_jaccard(span_sets[s1], span_sets[s2])

        df = pd.DataFrame(matrix, index=synonyms, columns=synonyms)
        df.to_excel(OUT_DIR / f"jaccard_matrix_{code}.xlsx", index=True)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Indice de Jaccard"})
        plt.title(f"Chevauchement des prédictions (par synonyme) – {code}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"heatmap_{code}.png")
        plt.close()

    print(f"\n Chevauchement par synonyme enregistré dans : {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
