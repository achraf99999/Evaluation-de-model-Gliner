# main.py

import sys
from pathlib import Path

# Ajouter le dossier src/ au chemin d'import
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.config import DEFAULT_JACCARD_THRESHOLD
from src.predict_by_synonym import main as run_predict_by_synonym
from src.predict_combinations import main as run_predict_combinations
from src.evaluate_union import evaluate_union
from src.evaluate_intersection import main as run_evaluate_intersection
from src.overlap_by_synonym import main as run_overlap_by_synonym
from src.overlap_combinations import main as run_overlap_combinations


def main():
    print(" Lancement du pipeline GLiNER\n")

    # Étape 1 : Prédiction synonyme par synonyme
    print("\n Étape 1 : Prédictions par synonyme")
    run_predict_by_synonym()

    # Étape 2 : Prédictions pour toutes les combinaisons de synonymes
    print("\n Étape 2 : Prédictions par combinaison de synonymes")
    run_predict_combinations()

    # Étape 3 : Évaluation des unions
    print("\n Étape 3 : Évaluation par UNION")
    evaluate_union(threshold=DEFAULT_JACCARD_THRESHOLD)

    # Étape 4 : Évaluation des intersections
    print("\n Étape 4 : Évaluation par INTERSECTION")
    run_evaluate_intersection()

    # Étape 5 : Recouvrement entre synonymes
    print("\n Étape 5 : Recouvrement entre prédictions (synonymes)")
    run_overlap_by_synonym()

    # Étape 6 : Recouvrement entre combinaisons
    print("\n Étape 6 : Recouvrement entre prédictions (combinaisons)")
    run_overlap_combinations()

    print("\n Pipeline GLiNER terminé avec succès !")

if __name__ == "__main__":
    main()
