import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from src.config import ENTITY_TYPES, PRED_COMBINATIONS_JSON, OVERLAP_DIR 
from src.utils import ensure_dir


# Répertoires
OUT_DIR = Path("outputs/overlap_analysis/synonym_COMBINATIONS")
# Le répertoire des combinaisons où les fichiers seront réellement sauvegardés
COMBINATIONS_OUT_DIR = OUT_DIR / "combinations"

# Assurez-vous que les deux répertoires existent
ensure_dir(OUT_DIR)
ensure_dir(COMBINATIONS_OUT_DIR) 

DEBUG_COMBINATIONS_FILE = PRED_COMBINATIONS_JSON


def compute_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def extract_combination_spans(debug_data, entity_code):
    """Récupère les spans prédits pour chaque combinaison de synonymes d'une entité."""
    combo_spans = {}
    for key, entries in debug_data.items():
        if not key.startswith(f"{entity_code}__"):
            continue
        combo_name = key.split("__", 1)[1]
        spans = {tuple(entry["span"]) for entry in entries}
        combo_spans[combo_name] = spans
    return combo_spans
    
def abbreviate_combo(combo_key: str) -> str:
    # Sépare les synonymes par "__" et prend la première lettre de chaque mot
    return "_".join(word[0].lower() for word in combo_key.split("__"))
def main():
    print(" Calcul des overlaps entre les prédictions des combinaisons de synonymes...")
    with open(DEBUG_COMBINATIONS_FILE, encoding="utf-8") as f:
        debug_data = json.load(f)

    for code in ENTITY_TYPES:
        print(f"\n Traitement de l'entité : {code}")
        spans_by_combo = extract_combination_spans(debug_data, code)
        combos = list(spans_by_combo.keys())
        n = len(combos)

        matrix = np.zeros((n, n))

        for i, j in combinations(range(n), 2):
            s1 = spans_by_combo[combos[i]]
            s2 = spans_by_combo[combos[j]]
            jaccard = compute_jaccard(s1, s2)
            matrix[i, j] = matrix[j, i] = jaccard

        for i in range(n):  # diagonale = 1.0
            matrix[i, i] = 1.0

        # Générer les noms de combinaisons abrégés pour les axes de la heatmap
        abbreviated_combos = [abbreviate_combo(c) for c in combos]
        df = pd.DataFrame(matrix, index=abbreviated_combos, columns=abbreviated_combos)

        # Sauvegarde CSV
        excel_path = COMBINATIONS_OUT_DIR / f"jaccard_matrix_{code}.xlsx"
        df.to_excel(excel_path) 
        print(f" Sauvegardé : {excel_path}")

        # Sauvegarde Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Indice de Jaccard'})
        plt.title(f"Recouvrement entre combinaisons de synonymes – {code}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        heatmap_path = COMBINATIONS_OUT_DIR / f"heatmap_{code}.png"
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        print(f" Sauvegardé : {heatmap_path}")

    print("\n Analyse de recouvrement terminée pour toutes les entités.")

if __name__ == "__main__":
    main()
