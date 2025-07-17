import json
import itertools
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from src.config import ENTITY_TYPES, OUTPUT_DIR, DEFAULT_JACCARD_THRESHOLD, DATA_PATH
from src.utils import prf1, load_json, load_corpus, jaccard

JACCARD_THRESHOLD = DEFAULT_JACCARD_THRESHOLD

def evaluate_individual_synonyms(debug_data, corpus):
    """
    Évalue les prédictions GLiNER pour chaque synonyme individuel d'une entité.
    Inclut le calcul des True Positives (TP), False Positives (FP), False Negatives (FN),
    et True Negatives (TN) pour chaque synonyme.
    """
    results = []
    corpus_map = {doc["text_id"]: doc for doc in corpus}

    for entity_code, synonyms in ENTITY_TYPES.items():
        # Calculer le nombre total de spans d'entités réelles dans le corpus
        # qui NE SONT PAS du type 'entity_code' actuel.
        # Cette valeur de TN est constante pour toutes les évaluations de synonymes sous la même entity_type.
        tn_for_current_entity_type = 0
        for doc_item in corpus:
            tn_for_current_entity_type += len({
                tuple(ent["spans"]) for ent in doc_item["entities"]
                if ent["code_entity"] != entity_code
            })

        for synonym_label in synonyms: # Boucle sur chaque synonyme individuel
            syn_key = f"{entity_code}__{synonym_label}" # Clé pour accéder aux prédictions de ce synonyme

            tp = fp = fn = 0 # Compteurs pour TP, FP, FN pour le synonyme actuel
            
            for doc in corpus:
                text_id = doc["text_id"]
                gold = {
                    tuple(ent["spans"])
                    for ent in doc["entities"]
                    if ent["code_entity"] == entity_code
                }

                pred_for_doc = {tuple(entry["span"]) for entry in debug_data.get(syn_key, []) if entry["text_id"] == text_id}

                matched_gold = set()
                matched_pred = set()

                # 1. Correspondances exactes (Exact Match)
                for span_p in pred_for_doc:
                    if span_p in gold:
                        tp += 1
                        matched_gold.add(span_p)
                        matched_pred.add(span_p)

                # 2. Correspondances partielles (Partial Match) sur les spans non appariés
                for span_p in pred_for_doc - matched_pred:  # Spans prédits non encore appariés
                    for span_g in gold - matched_gold:  # Spans gold non encore appariés
                        # Utilise la fonction jaccard de src.utils
                        iou = jaccard(set(range(*span_p)), set(range(*span_g)))
                        if iou >= JACCARD_THRESHOLD:
                            tp += 1
                            matched_gold.add(span_g)
                            matched_pred.add(span_p)
                            break # Une fois un match partiel trouvé pour span_p, on passe au suivant

                # Calcul des False Positives (FP) et False Negatives (FN)
                fp += len(pred_for_doc - matched_pred) # Spans prédits non appariés = faux positifs
                fn += len(gold - matched_gold)       # Spans gold non appariés = faux négatifs

            # Calcul de la précision, du rappel et du F1-score pour ce synonyme
            p, r, f1 = prf1(tp, fp, fn)
            results.append({
                "entity_type": entity_code,
                "synonym": synonym_label, # Utilise 'synonym' au lieu de 'combo'
                "precision": round(p, 6),
                "recall": round(r, 6),
                "f1": round(f1, 6),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn_for_current_entity_type
            })

    return pd.DataFrame(results)


# Fonction de plotting des métriques, adaptée pour les synonymes individuels
def save_metrics_and_plot_individual(df: pd.DataFrame, output_dir: Path, prefix: str):
    """
    Sauvegarde les métriques et génère des graphiques à barres pour les synonymes individuels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    excel_path = output_dir / f"metrics_{prefix}.xlsx"
    df.to_excel(excel_path, index=False)
    print(f" Metrics saved to: {excel_path}")

    for metric in ["precision", "recall", "f1"]:
        for code in df["entity_type"].unique():
            sub = df[df["entity_type"] == code].copy()
            sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)

            # --- MODIFICATION ICI: Utilisation des noms complets des synonymes ---
            base_display_labels = [s for s in sub["synonym"]] # Utilise le nom complet du synonyme
            
            final_display_labels = []
            seen_labels_count = defaultdict(int)
            for label in base_display_labels: # Itère sur les noms complets des synonymes
                seen_labels_count[label] += 1
                if seen_labels_count[label] > 1:
                    final_display_labels.append(f"{label}-{seen_labels_count[label]}")
                else:
                    final_display_labels.append(label)

            plt.figure(figsize=(14, 6))
            # Pas de logique de couleur basée sur la longueur de la combinaison ici (car ce sont des synonymes individuels)
            # Utilise la couleur par défaut de Matplotlib
            bars = plt.bar(final_display_labels, sub[metric])
            plt.xticks(rotation=0)
            plt.title(f"{metric.upper()} scores pour les différents synonymes de {code}")
            plt.ylim(0, 1.1)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                                 f"{height:.4f}", ha='center', va='bottom', fontsize=7)
            plt.tight_layout()
            plt.savefig(output_dir / f"{code}_{metric}_{prefix}_barplot.png", bbox_inches="tight")
            plt.close()

# Fonction de plotting de la matrice de confusion visuelle (inchangée, elle prend les valeurs directement)
def plot_confusion_matrix_visual(tp, fp, fn, tn, entity_type, label_name, output_path):
    """
    Génère et sauvegarde un plot visuel de la matrice de confusion au format 2x2.
    `label_name` peut être un synonyme individuel ou une combinaison.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    matrix_data = np.array([[0, 1], [1, 0]]) 

    cmap_colors = ['#d4edda', '#f8d7da']
    cmap = mcolors.ListedColormap(cmap_colors)

    ax.imshow(matrix_data, cmap=cmap)

    ax.text(0, 0, f"\n{tp}", ha="center", va="center", color="black", fontsize=16, weight="bold")
    ax.text(1, 0, f"\n{fp}", ha="center", va="center", color="black", fontsize=16, weight="bold")
    ax.text(0, 1, f"\n{fn}", ha="center", va="center", color="black", fontsize=16, weight="bold")
    ax.text(1, 1, f"\n{tn}", ha="center", va="center", color="black", fontsize=16, weight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=12)
    ax.set_yticklabels(["Positive", "Negative"], fontsize=12, rotation=90, va="center")

    ax.set_xlabel("True Class", fontsize=14, labelpad=20)
    ax.set_ylabel("Predicted Class", fontsize=14, labelpad=20, rotation=0, ha="right")

    # Titre mis à jour pour être plus générique (pour synonyme ou combo)
    ax.set_title(f"Matrice de Confusion pour {entity_type} - {label_name}", fontsize=16)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_individual_confusion_matrices_individual(df: pd.DataFrame, output_dir: Path, prefix: str):
    """
    Sauvegarde les TP, FP, FN, et TN pour chaque synonyme individuel dans des fichiers Excel,
    au format de matrice de confusion 2x2, et génère un plot visuel pour chacun.
    """
    confusion_matrix_dir = output_dir / "confusion_matrices_individual_synonyms" # Nouveau sous-répertoire
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving individual confusion matrices for individual synonyms to: {confusion_matrix_dir}")

    for index, row in df.iterrows():
        entity_type = row["entity_type"]
        synonym_name = row["synonym"] # Utilise 'synonym' au lieu de 'combo'
        tp = row["TP"]
        fp = row["FP"]
        fn = row["FN"]
        tn = row["TN"]

        # Création d'un DataFrame pour la matrice de confusion individuelle (format 2x2)
        data = {
            "Positive": [tp, fn],
            "Negative": [fp, tn]
        }
        confusion_df = pd.DataFrame(data, index=["Positive", "Negative"])
        confusion_df.columns.name = "True Class"
        confusion_df.index.name = "Predicted Class"

        # Création d'un nom de fichier unique et lisible pour Excel
        # Remplace les espaces et caractères non sûrs dans le nom du synonyme
        safe_label_name = synonym_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace("__", "_")
        excel_filename = f"confusion_matrix_data_{entity_type}_{safe_label_name}_{prefix}.xlsx"
        excel_file_path = confusion_matrix_dir / excel_filename
        
        confusion_df.to_excel(excel_file_path)
        
        # Création d'un nom de fichier unique et lisible pour l'image
        image_filename = f"confusion_matrix_plot_{entity_type}_{safe_label_name}_{prefix}.png"
        image_file_path = confusion_matrix_dir / image_filename
        
        # Appel de la fonction pour générer le plot visuel de la matrice de confusion
        plot_confusion_matrix_visual(tp, fp, fn, tn, entity_type, synonym_name, image_file_path)

        # print(f"  Saved Excel: {excel_file_path.name}, Plot: {image_file_path.name}")


def main_individual_evaluation():
    """Fonction principale pour l'évaluation des synonymes individuels."""
    print("\nChargement des données pour l'évaluation des synonymes individuels...")
    # Assurez-vous que 'debug_by_synonym.json' contient les prédictions
    # pour chaque synonyme individuel (e.g., "ENTITY__synonym_label").
    debug_data = load_json(OUTPUT_DIR / "debug_by_synonym.json")
    corpus = load_corpus(DATA_PATH)
    
    print("Évaluation des synonymes individuels...")
    df = evaluate_individual_synonyms(debug_data, corpus)
    
    prefix = "individual_synonyms" # Préfixe pour les noms de fichiers et titres spécifiques

    print("Sauvegarde des métriques et génération des graphiques pour les synonymes individuels...")
    save_metrics_and_plot_individual(df, OUTPUT_DIR / "results_individual_synonyms", prefix=prefix)
    
    # Appel de la fonction pour sauvegarder les matrices de confusion individuelles et leurs plots
    save_individual_confusion_matrices_individual(df, OUTPUT_DIR / "results_individual_synonyms", prefix=prefix)
    
    print("Évaluation des synonymes individuels terminée.")


if __name__ == "__main__":
    main_individual_evaluation()
