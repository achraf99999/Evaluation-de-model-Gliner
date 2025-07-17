import json
import itertools
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from src.config import ENTITY_TYPES, OUTPUT_DIR, DEFAULT_JACCARD_THRESHOLD, DATA_PATH
from src.utils import prf1, load_json, load_corpus # Assurez-vous que prf1, load_json, load_corpus sont bien dans utils.py

JACCARD_THRESHOLD = DEFAULT_JACCARD_THRESHOLD

def evaluate_intersection(debug_data, corpus):
    """
    Évalue l'intersection des prédictions GLiNER entre les synonymes pour chaque entité.
    Considère un span prédit s'il est présent dans l'ensemble des prédictions de TOUS les synonymes du combo.
    Inclut TP, FP, FN et TN (True Negatives, définis comme les spans d'entités réelles d'autres types).
    """
    results = []
    corpus_map = {doc["text_id"]: doc for doc in corpus}

    for code, synonyms in ENTITY_TYPES.items():
        spans_by_syn = defaultdict(lambda: defaultdict(set))

        # Charger les spans par synonyme
        for syn in synonyms:
            key = f"{code}__{syn}"
            for entry in debug_data.get(key, []):
                text_id = entry["text_id"]
                span = tuple(entry["span"])
                spans_by_syn[syn][text_id].add(span)
        tn_for_current_entity_type = 0
        for doc_item in corpus:
            tn_for_current_entity_type += len({
                tuple(ent["spans"]) for ent in doc_item["entities"]
                if ent["code_entity"] != code
            })

        # Calculer le nombre total de spans d'entités réelles qui NE SONT PAS du type 'code' actuel.
        # Ce sera notre valeur pour TN pour cette 'entity_type'.
        tn_for_current_entity_type = 0
        for doc_item in corpus:
            tn_for_current_entity_type += len({
                tuple(ent["spans"]) for ent in doc_item["entities"]
                if ent["code_entity"] != code
            })

        # Générer toutes les combinaisons de 2 à N synonymes
        for k in range(2, len(synonyms) + 1):
            for combo in itertools.combinations(synonyms, k):
                combo_key = "__".join(combo)
                spans_by_text = defaultdict(set)

                for text_id in corpus_map:
                    sets = [spans_by_syn[syn].get(text_id, set()) for syn in combo]
                    # Calcul de l'intersection des sets de spans
                    intersection = set.intersection(*sets) if sets else set()
                    spans_by_text[text_id] = intersection

                # Initialisation des compteurs pour les métriques de performance
                tp = fp = fn = 0
                for doc in corpus:
                    text_id = doc["text_id"]
                    # Récupération des spans 'gold standard' (vérité terrain) pour le type d'entité actuel
                    gold = {
                        tuple(ent["spans"])
                        for ent in doc["entities"]
                        if ent["code_entity"] == code
                    }
                    # Récupération des spans prédits (intersection)
                    pred = spans_by_text.get(text_id, set())

                    matched_gold = set()
                    matched_pred = set()

                    # Première passe: Correspondances exactes
                    for span in pred:
                        if span in gold:
                            tp += 1
                            matched_gold.add(span)
                            matched_pred.add(span)

                    # Deuxième passe: Correspondances partielles basées sur Jaccard Index
                    for span_p in pred - matched_pred:  # Spans prédits non encore appariés
                        for span_g in gold - matched_gold:  # Spans gold non encore appariés
                            # Calcul de l'IoU (Intersection over Union) ou Jaccard Index
                            iou = len(set(range(*span_p)) & set(range(*span_g))) / len(set(range(*span_p)) | set(range(*span_g)))
                            if iou >= JACCARD_THRESHOLD:
                                tp += 1
                                matched_gold.add(span_g)
                                matched_pred.add(span_p)
                                break # On passe au prochain span prédit si un match est trouvé

                    # Calcul des faux positifs et faux négatifs
                    fp += len(pred - matched_pred) # Spans prédits non appariés = faux positifs
                    fn += len(gold - matched_gold) # Spans gold non appariés = faux négatifs

                p, r, f1 = prf1(tp, fp, fn)
                results.append({
                    "entity_type": code,
                    "combo": combo_key,
                    "precision": round(p, 6),
                    "recall": round(r, 6),
                    "f1": round(f1, 6),
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn_for_current_entity_type # Ajout de TN
                })

    return pd.DataFrame(results)


def abbreviate_combo(combo_key: str) -> str:
    """Abbréviation d'une combinaison : 'anatomy__body part' → 'a_b'."""
    # Sépare les synonymes par "__" et prend la première lettre de chaque mot
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


def save_metrics_and_plot(df: pd.DataFrame, output_dir: Path, prefix: str):
    """
    Sauvegarde les métriques et génère des graphiques.
    :param df: DataFrame contenant les métriques d'évaluation.
    :param output_dir: Répertoire de sortie pour les fichiers.
    :param prefix: Préfixe pour les noms de fichiers et les titres de graphiques (ex: 'set_intersection', 'set_union').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde de toutes les métriques dans un fichier Excel principal
    excel_path = output_dir / f"metrics_{prefix}.xlsx"
    df.to_excel(excel_path, index=False)
    print(f" Metrics saved to: {excel_path}")

    for metric in ["precision", "recall", "f1"]: # Note: TP, FP, FN, TN sont dans le .xlsx mais ne sont pas tracés ici
        for code in df["entity_type"].unique():
            sub = df[df["entity_type"] == code].copy()
            sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)

            # Traitement pour s'assurer que les labels des barres sont uniques
            raw_abbreviated_labels = [abbreviate_combo(c) for c in sub["combo"]]
            final_display_labels = []
            seen_labels_count = defaultdict(int)
            for label in raw_abbreviated_labels:
                seen_labels_count[label] += 1
                if seen_labels_count[label] > 1:
                    final_display_labels.append(f"{label}-{seen_labels_count[label]}")
                else:
                    final_display_labels.append(label)

            plt.figure(figsize=(14, 6))
            colors = [
                "green" if len(c.split("__")) == 2 else
                "orange" if len(c.split("__")) == 3 else
                "red" if len(c.split("__")) == 4 else
                "blue" if len(c.split("__")) == 5 else
                "purple"
                for c in sub["combo"]
            ]

            bars = plt.bar(final_display_labels, sub[metric], color=colors)
            plt.xticks(rotation=90)
            # Titre ajusté pour inclure le préfixe
            plt.title(f"{metric.upper()} scores pour les ensembles d'intersection des différents synonymes de {code}")
            plt.ylim(0, 1.1)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                                 f"{height:.4f}", ha='center', va='bottom', fontsize=7)
            plt.tight_layout()
            # Nom de fichier ajusté pour inclure le préfixe
            plt.savefig(output_dir / f"{code}_{metric}_{prefix}_barplot.png", bbox_inches="tight")
            plt.close()

def plot_confusion_matrix_visual(tp, fp, fn, tn, entity_type, combo_name, output_path):
    """
    Génère et sauvegarde un plot visuel de la matrice de confusion au format 2x2.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Créer une matrice factice pour imshow (les couleurs sont basées sur une simple correspondance de valeurs)
    # 0 pour vert (TP, TN), 1 pour rouge (FP, FN)
    matrix_data = np.array([[0, 1], [1, 0]]) 

    # Définir des couleurs personnalisées pour le plot
    # Vert clair pour les classifications correctes, Rouge clair pour les incorrectes
    cmap_colors = ['#d4edda', '#f8d7da'] # Couleurs Bootstrap 'success' et 'danger'
    cmap = mcolors.ListedColormap(cmap_colors)

    ax.imshow(matrix_data, cmap=cmap)

    # Définir le texte des cellules avec les valeurs réelles
    ax.text(0, 0, f"\n{tp}", ha="center", va="center", color="black", fontsize=16, weight="bold")
    ax.text(1, 0, f"\n{fp}", ha="center", va="center", color="black", fontsize=16, weight="bold")
    ax.text(0, 1, f"\n{fn}", ha="center", va="center", color="black", fontsize=16, weight="bold")
    ax.text(1, 1, f"\n{tn}", ha="center", va="center", color="black", fontsize=16, weight="bold")

    # Définir les étiquettes des axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=12)
    ax.set_yticklabels(["Positive", "Negative"], fontsize=12, rotation=90, va="center")

    ax.set_xlabel("True Class", fontsize=14, labelpad=20)
    ax.set_ylabel("Predicted Class", fontsize=14, labelpad=20, rotation=0, ha="right")

    ax.set_title(f"Matrice de Confusion pour l'ensemble d'intersection \n des synonyms { {combo_name} }de \n type {entity_type}  ", fontsize=16)

    # Ajuster la grille et les bordures pour correspondre au style d'une matrice
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1.5) # Lignes de grille plus visibles
    ax.tick_params(which='minor', bottom=False, left=False) # Supprimer les marques des ticks mineurs
    
    # Assurer que les bordures de la matrice sont visibles
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


def save_individual_confusion_matrices(df: pd.DataFrame, output_dir: Path, prefix: str):
    """
    Sauvegarde les TP, FP, FN, et TN pour chaque combinaison dans des fichiers Excel individuels,
    au format de matrice de confusion 2x2, et génère un plot visuel pour chacun.
    """
    confusion_matrix_dir = output_dir / "confusion_matrices" # Sous-répertoire dédié
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving individual confusion matrices to: {confusion_matrix_dir}")

    for index, row in df.iterrows():
        entity_type = row["entity_type"]
        combo = row["combo"]
        tp = row["TP"]
        fp = row["FP"]
        fn = row["FN"]
        tn = row["TN"] # Récupération de TN

        # Création d'un DataFrame pour la matrice de confusion individuelle (format 2x2)
        data = {
            "Positive": [tp, fn],
            "Negative": [fp, tn]
        }
        confusion_df = pd.DataFrame(data, index=["Positive", "Negative"])
        confusion_df.columns.name = "True Class"
        confusion_df.index.name = "Predicted Class"

        # Création d'un nom de fichier unique et lisible pour Excel
        safe_combo_name = combo.replace("__", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")
        excel_filename = f"confusion_matrix_data_{entity_type}_{safe_combo_name}_{prefix}.xlsx"
        excel_file_path = confusion_matrix_dir / excel_filename
        
        confusion_df.to_excel(excel_file_path)
        
        # Création d'un nom de fichier unique et lisible pour l'image
        image_filename = f"confusion_matrix_plot_{entity_type}_{safe_combo_name}_{prefix}.png"
        image_file_path = confusion_matrix_dir / image_filename
        
        # Appel de la fonction pour générer le plot visuel de la matrice de confusion
        plot_confusion_matrix_visual(tp, fp, fn, tn, entity_type, combo, image_file_path)

        # print(f"  Saved Excel: {excel_file_path.name}, Plot: {image_file_path.name}") # Décommenter pour voir chaque fichier sauvegardé


def main_intersection():
    """Fonction principale pour l'évaluation basée sur l'intersection."""
    print("Chargement des données pour l'évaluation de l'intersection...")
    debug_data = load_json(OUTPUT_DIR / "debug_by_synonym.json")
    corpus = load_corpus(DATA_PATH)

    print("Évaluation de l'intersection des prédictions...")
    df = evaluate_intersection(debug_data, corpus)
    
    # Préfixe pour les noms de fichiers et titres
    prefix = "set_intersection"

    print("Sauvegarde des métriques et génération des graphiques pour l'intersection...")
    save_metrics_and_plot(df, OUTPUT_DIR / "results_intersection", prefix=prefix)
    
    # Appel de la nouvelle fonction pour sauvegarder les matrices de confusion individuelles et leurs plots
    save_individual_confusion_matrices(df, OUTPUT_DIR / "results_intersection", prefix=prefix)
    
    print("Évaluation de l'intersection terminée.")


# Note: La fonction evaluate_union (si elle existe et est commentée),
# aura une structure similaire pour ses métriques (precision_exact, etc.)
# Si vous la décommentez, assurez-vous qu'elle utilise aussi to_excel.


if __name__ == "__main__":
    # Exécute la fonction principale pour l'évaluation basée sur l'intersection
    main_intersection()
    
    