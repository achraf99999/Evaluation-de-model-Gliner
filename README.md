#  Projet d'Évaluation et d'Amélioration du Modèle GLiNER

[![Python](https://img.shields.io/badge/Python-3.9.11+-blue.svg)](https://www.python.org/downloads/)
[![GLiNER](https://img.shields.io/badge/GLiNER-Model-green.svg)](https://github.com/urchade/GLiNER)

##  Description

Ce projet présente une **évaluation complète et des méthodologies d'amélioration** pour le modèle GLiNER (Generalist and Lightweight model for Named Entity Recognition). Il comprend des analyses quantitatives et qualitatives approfondies, des techniques d'amélioration du modèle, et la construction d'une base de données de connaissances enrichie.

###  Objectifs Principaux

- **Évaluation multi-dimensionnelle** du modèle GLiNER avec différentes stratégies de synonymes
- **Analyse comparative** des performances (intersection vs union des prédictions)
- **Amélioration du modèle** via fine-tuning et changement d'encodeur
- **Construction d'une base de données** de connaissances enrichie par LLM
- **Documentation complète** des méthodologies et résultats

##  Structure du Projet

```
📁 Evaluation-de-model-Gliner/
├── 📁 src/                          # Pipeline d'évaluation principal
│   ├── 📄 config.py                 # Configuration globale
│   ├── 📄 utils.py                  # Fonctions utilitaires
│   ├── 📄 evaluate_intersection.py  # Évaluation par intersection
│   ├── 📄 evaluate_union.py         # Évaluation par union
│   ├── 📄 evaluate_union_indiv.py   # Évaluation union individuelle
│   ├── 📄 predict_by_synonym.py     # Prédiction par synonymes
│   ├── 📄 predict_combinations.py   # Prédiction par combinaisons
│   ├── 📄 overlap_by_synonym.py     # Analyse chevauchement synonymes
│   └── 📄 overlap_combinations.py   # Analyse chevauchement combinaisons
├── 📁 data/                         # Données d'entrée
│   └── 📄 fulldata.json            # Dataset principal
│   └── 📄 BioASQ_BIONNE_test_2024.zip
│   └── 📄 BioASQ_BIONNE_training_2024.zip
│   └── 📄 Fine-Tuning-dataset-def.json
│   └── 📄 Fine-Tuning-dataset-entity.json
├── 📁 data_factory/                 # Transformation des données
│   └── 📄 dataloader-veritée-terrain.py                  # Scripts de préparation
│   └── 📄 entitie-to-def.py
│   └── 📄 fndata.py
├── 📁 outputs/                      # Résultats et métriques
│   ├── 📄 debug_by_synonym.json    # Prédictions brutes (synonymes)
│   ├── 📄 debug_combinations.json  # Prédictions brutes (combinaisons)
│   ├── 📄 mlm_synonym_prediction_results.xlsx # Prédictions brutes (combinaisons)
│   ├── 📁 results_intersection/     # Métriques intersection
│   ├── 📁 results_union/           # Métriques union
│   ├── 📁 entity_traces/           # Métriques union
│   └── 📁 overlap_analysis/        # Analyses de chevauchement
├── 📁 Conception_de_BD/            # Base de données de connaissances
│   └── 📄 build_kb.py
│   └── 📄 enrich_kb_with_llm.py
│   └── 📄 knowledge_base.json
│   └── 📄 knowledge_base_enriched.json

├── 📁 changement-encodeur/         # Modification d'encodeur
│   ├── 📄 GLiNER-MPnet.py
│   ├── 📄 GlinerJina.py
│   └── 📄 extract-weights-GLiNER.py
│   └── 📄 main.py
├── 📄 fine-tuning.ipynb           # Fine-tuning du modèle
├── 📄 main.evaluation.py          # Pipeline d'évaluation complet
├── 📄 requirements.txt            # Dépendances Python
└── 📄 Rapport de fin d'année.pdf  # Rapport technique détaillé
```

##  Composants Principaux

### 1. Pipeline d'Évaluation (`src/`)

#### **Configuration et Utilitaires**
- **`config.py`** : Configuration centralisée (entités, synonymes, seuils, chemins)
- **`utils.py`** : Fonctions de base (chargement données, calcul métriques, indice Jaccard)

#### **Prédiction d'Entités**
- **`predict_by_synonym.py`** : Prédictions avec synonymes individuels
- **`predict_combinations.py`** : Prédictions avec combinaisons de synonymes

#### **Évaluation des Performances**
- **`evaluate_intersection.py`** : Évaluation basée sur l'intersection des prédictions
- **`evaluate_union.py`** : Évaluation basée sur l'union des prédictions
- **`evaluate_union_indiv.py`** : Contribution individuelle à l'union

#### **Analyse des Chevauchements**
- **`overlap_by_synonym.py`** : Matrices de Jaccard pour synonymes
- **`overlap_combinations.py`** : Matrices de Jaccard pour combinaisons

### 2.  Base de Données de Connaissances (`Conception_de_BD/`)

Construction d'une **knowledge database** enrichie comprenant :
- Relations entre codes d'entités et synonymes
- Enrichissement automatique via **Gemini Flash 2.0**
- Structure relationnelle optimisée pour les requêtes NER

### 3.  Amélioration du Modèle (`changement-encodeur/`)

#### **Extraction des Poids GLiNER**
- Algorithmes d'extraction des poids des différentes parties du modèle
- Analyse des composants internes de GLiNER

#### **Changement d'Encodeur**
- **Expérimentation 1** : Remplacement par **MPNet**
- **Expérimentation 2** : Remplacement par **Jina**
- Évaluation comparative des performances

### 4.  Fine-Tuning (`fine-tuning.ipynb`)

Notebook complet pour le fine-tuning du modèle GLiNER :
- Préparation des données d'entraînement
- Configuration des hyperparamètres
- Processus d'entraînement optimisé
- Validation et évaluation des performances

### 5.  Factory de Données (`data_factory/`)

Scripts de transformation et préparation des données :
- Struturation de data  pour l'évaluation
- Préparation pour le fine-tuning
- Validation et nettoyage des datasets

## 📈 Métriques d'Évaluation

### **Métriques Principales**
- **Précision** : Exactitude des prédictions
- **Rappel** : Couverture des entités réelles
- **Score F1** : Moyenne harmonique précision/rappel

### **Méthodes d'Évaluation**
- **Correspondances exactes** : Matches parfaits des spans
- **Correspondances partielles** : Utilisation du seuil de Jaccard
- **Analyse intersection/union** : Stratégies de combinaison des prédictions

### **Analyses de Chevauchement**
- **Indice de Jaccard** : Mesure de similarité entre prédictions
- **Analyses par synonymes** : Évaluation individuelle 

## 🔬 Méthodologies d'Amélioration

### **1. Optimisation des Synonymes**
- Analyse de l'impact des différents synonymes
- Stratégies de combinaison optimales
- Identification des synonymes les plus performants

### **2. Modification Architecturale**
- Changement d'encodeur (MPNet, Jina)
- Extraction et analyse des poids
- Évaluation comparative des architectures

### **3. Fine-Tuning Spécialisé**
- Adaptation aux domaines spécifiques
- Optimisation des hyperparamètres
- Validation croisée rigoureuse

### **4. Enrichissement par LLM**
- Génération de relations sémantiques(entre les code d'entitée et les synonymes ) 
- Enrichissement  de la base de connaissances
- Etablissement d'un architucture GraphRAG( à faire )  

##  Résultats et Sorties

### **Fichiers de Prédictions**
- `outputs/debug_by_synonym.json` : Prédictions brutes par synonymes
- `outputs/debug_combinations.json` : Prédictions par combinaisons

### **Métriques d'Évaluation**
- `outputs/results_intersection/` : Métriques et graphiques intersection
- `outputs/results_union/` : Métriques et graphiques union

### **Analyses de Chevauchement**
- `outputs/overlap_analysis/synonym_level/` : Matrices Jaccard synonymes
- `outputs/overlap_analysis/synonym_COMBINATIONS/` : Matrices combinaisons

## 🛠️ Installation et Utilisation

### **Prérequis**
```bash
Python 3.9.11
pip install -r requirements.txt
```



### **Utilisation**

#### **1. Évaluation Complète**
```bash
python main.evaluation.py
```

#### **2. Prédictions Individuelles**
```bash
# Prédictions par synonymes
python src/predict_by_synonym.py

# Prédictions par combinaisons
python src/predict_combinations.py
```

#### **3. Analyses Spécifiques**
```bash
# Évaluation intersection
python src/evaluate_intersection.py

# Évaluation union
python src/evaluate_union.py

# Analyse chevauchements
python src/overlap_by_synonym.py
```

#### **4. Fine-Tuning**
```bash
jupyter notebook fine-tuning.ipynb
```

## 📚 Documentation Technique

### **Rapport Complet**
Le fichier `Rapport de fin d'année.pdf` contient :

#### **État de l'Art**
- Revue complète des modèles NER
- Positionnement de GLiNER
- Comparaisons avec les autres modèles de NER.

#### **Architecture GLiNER**
- Décomposition fonctionnelle détaillée
- Analyse des composants internes

#### **Méthodologies d'Évaluation**
- Protocoles  d'Évaluation
- Validation des métriques
- Stratégies de test robustes

#### **Analyses Quantitatives**
- Résultats détaillés par métrique
- Comparaisons statistiques

#### **Analyses Qualitatives**
- Étude des erreurs
- Cas d'usage spécifiques
- Recommandations d'amélioration

#### **Méthodologies d'Amélioration**
- Stratégies de fine-tuning
- Enrichissement sémantique


##  Contribution

Les contributions sont les bienvenues ! Merci de :
1. Fork le projet
2. Créer une branche feature
3. Commiter vos changements
4. Ouvrir une Pull Request


## 👨‍💻 Auteur

**Achraf** - [GitHub](https://github.com/achraf99999)

---

*Ce projet a été développé dans le cadre d'un stage de recherche sur l'amélioration des modèles de reconnaissance d'entités nommées à l'ISIS sous encadremant de M.Yohann Chasseray .*
