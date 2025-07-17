Ce README fournit un aperçu de la structure du projet et de la fonction de chaque fichier.

## Structure du Projet

Le projet est organisé comme suit :

.
├── src/
│ ├── config.py
│ ├── utils.py
│ ├── evaluate_intersection.py
│ ├── evaluate_union.py
│ ├── evaluate_union_indiv.py
│ ├── predict_by_synonym.py
│ ├── predict_combinations.py
│ ├── overlap_by_synonym.py
│ └── overlap_combinations.py
├── data/
│ └── fulldata.json
├── dataf_factory/
│ └── data.py
├── outputs/
│ ├── debug_by_synonym.json
│ ├── debug_combinations.json
│ ├── results_intersection/
│ │ └── ... (métriques(csv) et graphiques pour l'intersection(png))
│ ├── results_union/
│ │ └── ... (métriques(csv) et graphiques pour l'union)
│ └── overlap_analysis/
│   	 ├── synonym_level/
│   	 │        └── ... (matrices(csv) et matrices de Jaccard pour le chevauchement par synonyme(png))
│   	 └── synonym_COMBINATIONS/
│             		└── combinations/
│                      		└── ... (matrices et cartes de chaleur de Jaccard pour le chevauchement par combinaison)
└── README.md


## Description des Fichiers

Voici une brève explication de chaque fichier important :

* **`src/config.py`** : Ce fichier centralise toutes les configurations globales du projet. Il définit les types d'entités et leurs synonymes, les chemins d'accès aux fichiers d'entrée et de sortie, le nom du modèle GLiNER, et les seuils par défaut pour la prédiction et la similarité de Jaccard.

* **`src/utils.py`** : Ce fichier  contient des fonctions d'aide utilisées par les différents scripts. Il inclut des fonctions pour charger des données JSON, charger le modèle GLiNER, calculer l'indice de Jaccard entre des ensembles, calculer la précision, le rappel et le score F1.

* **`src/predict_by_synonym.py`** : Ce script utilise le modèle GLiNER pour prédire des entités pour chaque synonyme individuel défini dans `config.py`. Les prédictions, incluant l'`text_id`, le segment (span), le texte de l'entité et l'étiquette, sont sauvegardées dans `outputs/debug_by_synonym.json`.

* **`src/predict_combinations.py`** : Similaire à `predict_by_synonym.py`, ce script prédit des entités en utilisant des combinaisons de synonymes. Il itère sur toutes les combinaisons possibles de synonymes pour chaque type d'entité et sauvegarde les prédictions dans `outputs/debug_combinations.json`.

* **`src/evaluate_union.py`** : Ce script évalue la performance des prédictions d'entités basées sur l'*union* des segments (spans) prédits par différentes combinaisons de synonymes. Il calcule la précision, le rappel et le score F1 (correspondances exactes et partielles utilisant un seuil de Jaccard) pour chaque combinaison, sauvegarde les résultats dans `outputs/results_union/metrics_set_union.csv` et génère des graphiques à barres.

* **`src/evaluate_union_indiv.py`** : Ce script évalue également l'union des prédictions, en se concentrant spécifiquement sur la contribution des prédictions de synonymes individuels à l'union globale. Il calcule la précision, le rappel et le score F1, et sauvegarde les métriques et les graphiques dans le répertoire `results_union`.

* **`src/evaluate_intersection.py`** : Ce script évalue la performance des prédictions d'entités basées sur l'*intersection* des segments (spans) prédits par différentes combinaisons de synonymes. Il calcule la précision, le rappel et le score F1 (prenant en compte les correspondances exactes et partielles avec un seuil de Jaccard), sauvegarde les résultats dans `outputs/results_intersection/metrics_set_intersections.csv` et génère des graphiques à barres.

* **`src/overlap_by_synonym.py`** : Ce script analyse le chevauchement (en utilisant l'indice de Jaccard) entre les prédictions d'entités faites par des *synonymes individuels* pour chaque type d'entité. Il génère une matrice de similarité de Jaccard et une carte de chaleur, les sauvegardant dans `outputs/overlap_analysis/synonym_level/`.

* **`src/overlap_combinations.py`** : Ce script calcule l'indice de Jaccard pour mesurer le chevauchement entre les prédictions dérivées de *différentes combinaisons de synonymes* pour chaque type d'entité. Il produit une matrice de Jaccard et une carte de chaleur dans `outputs/overlap_analysis/synonym_COMBINATIONS/combinations/`.

* **`data/fulldata.json`** : Ce répertoire contient l'ensemble de données d'entrée utilisé pour la prédiction et l'évaluation des entités.
* **`data_factoy/data.py`** : Ce répertoire contient l'ensemble de données d'entrée utilisé pour la prédiction et l'évaluation des entités.

* **`outputs/`** : Ce répertoire stocke tous les fichiers de sortie générés, y compris :
    * `debug_by_synonym.json` : Prédictions brutes pour chaque synonyme individuel.
    * `debug_combinations.json` : Prédictions brutes pour chaque combinaison de synonymes.
    * `results_intersection/` : Contient les fichiers CSV des métriques d'évaluation et les graphiques à barres pour les évaluations basées sur l'intersection.
    * `results_union/` : Contient les fichiers CSV des métriques d'évaluation et les graphiques à barres pour les évaluations basées sur l'union.
    * `overlap_analysis/` : Contient des sous-répertoires avec des matrices de Jaccard et des cartes de chaleur illustrant le chevauchement entre les prédictions au niveau des synonymes et des combinaisons.
