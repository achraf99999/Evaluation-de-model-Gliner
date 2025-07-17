import json
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.config import ENTITY_TYPES, DATA_PATH, PRED_COMBINATIONS_JSON, UNION_DIR ,DEFAULT_JACCARD_THRESHOLD , OUTPUT_DIR

from src.utils import load_json, jaccard, prf1

OUTPUT_UNION_DIR=UNION_DIR
def evaluate_union(threshold=0.5):
    print(f" Évaluation par union (Jaccard threshold = {threshold})")
    debug_data = load_json(PRED_COMBINATIONS_JSON)
    corpus = load_json(DATA_PATH)
    corpus_map = {doc["text_id"]: doc for doc in corpus}
    
    results = []

    for entity_code, synonyms in ENTITY_TYPES.items():
        for k in range(2, len(synonyms) + 1):
            for combo in itertools.combinations(synonyms, k):
                combo_key = "__".join(combo)
                debug_key = f"{entity_code}__{combo_key}"

                spans_by_text = defaultdict(set)
                for entry in debug_data.get(debug_key, []):
                    spans_by_text[entry["text_id"]].add(tuple(entry["span"]))

                tp_ex = fp_ex = fn_ex = 0
                tp_pa = fp_pa = fn_pa = 0

                for doc in corpus:
                    gold = {
                        tuple(ent["spans"])
                        for ent in doc["entities"]
                        if ent["code_entity"] == entity_code
                    }
                    pred = spans_by_text.get(doc["text_id"], set())

                    matched_gold = set()
                    matched_pred = set()

                    # Exact match
                    for span in pred:
                        if span in gold:
                            tp_ex += 1
                            matched_gold.add(span)
                            matched_pred.add(span)

                    # Partial match (only on unmatched)
                    for span_p in pred - matched_pred:
                        for span_g in gold - matched_gold:
                            if jaccard(set(range(*span_p)), set(range(*span_g))) >= threshold:
                                tp_pa += 1
                                matched_gold.add(span_g)
                                matched_pred.add(span_p)
                                break

                    fp_ex += len(pred - matched_pred)
                    fn_ex += len(gold - matched_gold)

                    fp_pa += len(pred - matched_pred)
                    fn_pa += len(gold - matched_gold)

                # Compute metrics
                p1, r1, f1_1 = prf1(tp_ex, fp_ex, fn_ex)
                p2, r2, f1_2 = prf1(tp_ex + tp_pa, fp_pa, fn_pa)

                results.append({
                    "entity_type": entity_code,
                    "combo": combo_key,
                    "precision_exact": round(p1, 4),
                    "recall_exact": round(r1, 4),
                    "f1_exact": round(f1_1, 4),
                    "precision_partial": round(p2, 4),
                    "recall_partial": round(r2, 4),
                    "f1_partial": round(f1_2, 4)
                })

    df = pd.DataFrame(results)
    OUTPUT_UNION_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_UNION_DIR / "metrics_set_union.xlsx"
    df.to_excel(csv_path, index=False)
    print(f" Metrics saved to: {csv_path}")

    plot_metrics(df, OUTPUT_UNION_DIR)


def abbreviate_combo(combo_key: str) -> str:
    """
    Abbréviation d'une combinaison : 'synonym1__synonym2' → 's1_s2'.
    Gère également les noms de synonymes multi-mots (ex: 'body part' -> 'bp').
    """
    individual_synonyms = combo_key.split("__")
    abbreviated_parts = []
    for syn_part in individual_synonyms:
        words = syn_part.split() # Sépare les mots dans le nom du synonyme
        if words:
            # Prend la première lettre de chaque mot dans le nom du synonyme et les joint
            abbreviated_parts.append("".join(word[0].lower() for word in words))
        else:
            abbreviated_parts.append("") # Gère les parties de chaîne vides si elles existent

    return "_".join(abbreviated_parts)


def plot_metrics(df, output_dir):
    metrics = ["precision_exact", "recall_exact", "f1_exact", "precision_partial", "recall_partial", "f1_partial"]
    for entity in df["entity_type"].unique():
        sub = df[df["entity_type"] == entity]
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sub_sorted = sub.sort_values(metric, ascending=False)
            raw_abbreviated_labels = [abbreviate_combo(c) for c in sub_sorted["combo"]]
            
            # Traitement pour rendre les labels uniques en cas de collision d'abréviations
            final_display_labels = []
            seen_labels_count = defaultdict(int)
            for label in raw_abbreviated_labels:
                seen_labels_count[label] += 1
                if seen_labels_count[label] > 1:
                    # Si cette abréviation a déjà été vue, ajoute un suffixe numérique
                    final_display_labels.append(f"{label}-{seen_labels_count[label]}")
                else:
                    final_display_labels.append(label)            
            colors = [
                "green" if len(combo.split("__")) == 2 else
                "orange" if len(combo.split("__")) == 3 else
                "red" if len(combo.split("__")) == 4 else
                "blue" if len(combo.split("__")) == 5 else
                "purple"
                for combo in sub_sorted["combo"]
            ]
            bars = plt.bar(final_display_labels, sub_sorted[metric], color=colors)
            plt.xticks(rotation=90)
            plt.title(f"{metric} – {entity}")
            plt.ylim(0, 1.1)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                         ha='center', va='bottom', fontsize=7)
            plt.tight_layout()
            fname = output_dir / f"{entity}_{metric}_set_union_barplot.png"
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
            print(f" Saved plot: {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="Jaccard threshold for partial match")
    args = parser.parse_args()

    evaluate_union(threshold=args.threshold)
