# Projet d'Évaluation d'Entités GLiNER

Ce README fournit un aperçu de la structure du projet et de la fonction de chaque fichier.

## Structure du Projet

Le projet est organisé comme suit :

```
.
├── src/
│   ├── config.py
│   ├── utils.py
│   ├── evaluate_intersection.py
│   ├── evaluate_union.py
│   ├── evaluate_union_indiv.py
│   ├── predict_by_synonym.py
│   ├── predict_combinations.py
│   ├── overlap_by_synonym.py
│   └── overlap_combinations.py
├── data/
│   └── fulldata.json
├── dataf_factory/
│   └── data.py
├── outputs/
│   ├── debug_by_synonym.json
│   ├── debug_combinations.json
│   ├── results_intersection/
│   │   └── ... (métriques(csv) et graphiques pour l'intersection(png))
│   ├── results_union/
│   │   └── ... (métriques(csv) et graphiques pour l'union)
│   └── overlap_analysis/
│       ├── synonym_level/
│       │   └── ... (matrices(csv) et matrices de Jaccard pour le chevauchement par synonyme(png))
│       └── synonym_COMBINATIONS/
│           └── combinations/
│               └── ... (matrices et cartes de chaleur de Jaccard pour le chevauchement par combinaison)
└── README.md
```

## Description des Fichiers

### 📁 Répertoire `src/`

#### `config.py`
Centralise toutes les configurations globales du projet. Il définit :
- Les types d'entités et leurs synonymes
- Les chemins d'accès aux fichiers d'entrée et de sortie
- Le nom du modèle GLiNER
- Les seuils par défaut pour la prédiction et la similarité de Jaccard

#### `utils.py`
Contient des fonctions d'aide utilisées par les différents scripts :
- Chargement des données JSON
- Chargement du modèle GLiNER
- Calcul de l'indice de Jaccard entre des ensembles
- Calcul de la précision, du rappel et du score F1

#### `predict_by_synonym.py`
Utilise le modèle GLiNER pour prédire des entités pour chaque synonyme individuel défini dans `config.py`. Les prédictions incluent :
- `text_id`
- Segment (span)
- Texte de l'entité
- Étiquette

**Sortie :** `outputs/debug_by_synonym.json`

#### `predict_combinations.py`
Similaire à `predict_by_synonym.py`, ce script prédit des entités en utilisant des combinaisons de synonymes. Il itère sur toutes les combinaisons possibles de synonymes pour chaque type d'entité.

**Sortie :** `outputs/debug_combinations.json`

#### `evaluate_union.py`
Évalue la performance des prédictions d'entités basées sur l'**union** des segments (spans) prédits par différentes combinaisons de synonymes. 

**Métriques calculées :**
- Précision
- Rappel
- Score F1 (correspondances exactes et partielles utilisant un seuil de Jaccard)

**Sorties :**
- `outputs/results_union/metrics_set_union.csv`
- Graphiques à barres

#### `evaluate_union_indiv.py`
Évalue également l'union des prédictions, en se concentrant spécifiquement sur la contribution des prédictions de synonymes individuels à l'union globale.

**Sorties :** Métriques et graphiques dans le répertoire `results_union`

#### `evaluate_intersection.py`
Évalue la performance des prédictions d'entités basées sur l'**intersection** des segments (spans) prédits par différentes combinaisons de synonymes.

**Métriques calculées :**
- Précision
- Rappel
- Score F1 (correspondances exactes et partielles avec un seuil de Jaccard)

**Sorties :**
- `outputs/results_intersection/metrics_set_intersections.csv`
- Graphiques à barres

#### `overlap_by_synonym.py`
Analyse le chevauchement (en utilisant l'indice de Jaccard) entre les prédictions d'entités faites par des **synonymes individuels** pour chaque type d'entité.

**Sorties :**
- Matrice de similarité de Jaccard
- Carte de chaleur
- Répertoire : `outputs/overlap_analysis/synonym_level/`

#### `overlap_combinations.py`
Calcule l'indice de Jaccard pour mesurer le chevauchement entre les prédictions dérivées de **différentes combinaisons de synonymes** pour chaque type d'entité.

**Sorties :**
- Matrice de Jaccard
- Carte de chaleur
- Répertoire : `outputs/overlap_analysis/synonym_COMBINATIONS/combinations/`

### 📁 Répertoire `data/`

#### `fulldata.json`
Ensemble de données d'entrée utilisé pour la prédiction et l'évaluation des entités.

### 📁 Répertoire `dataf_factory/`

#### `data.py`
Script de traitement des données d'entrée pour la prédiction et l'évaluation des entités.

### 📁 Répertoire `outputs/`

Ce répertoire stocke tous les fichiers de sortie générés :

- **`debug_by_synonym.json`** : Prédictions brutes pour chaque synonyme individuel
- **`debug_combinations.json`** : Prédictions brutes pour chaque combinaison de synonymes
- **`results_intersection/`** : Fichiers CSV des métriques d'évaluation et graphiques à barres pour les évaluations basées sur l'intersection
- **`results_union/`** : Fichiers CSV des métriques d'évaluation et graphiques à barres pour les évaluations basées sur l'union
- **`overlap_analysis/`** : Sous-répertoires avec des matrices de Jaccard et des cartes de chaleur illustrant le chevauchement entre les prédictions au niveau des synonymes et des combinaisons

## 🚀 Utilisation

[Ajoutez ici les instructions d'installation et d'utilisation du projet]

## 📋 Prérequis

[Listez ici les dépendances et prérequis nécessaires]

## 🤝 Contribution

[Ajoutez ici les guidelines pour contribuer au projet]

## 📄 Licence

[Ajoutez ici les informations de licence]
