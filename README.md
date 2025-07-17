🚀 Aperçu du Projet : Analyse de l'Extraction d'Entités avec GLiNER
Ce dépôt contient un projet d'analyse de l'extraction d'entités, exploitant le modèle GLiNER. Il explore différentes stratégies de prédiction (par synonyme individuel et par combinaisons de synonymes) et évalue leur performance via l'union et l'intersection des segments prédits. Le projet analyse également le chevauchement entre ces prédictions.

📂 Structure du Projet
Le projet est organisé de manière logique pour faciliter la navigation et la compréhension :

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
│   │   └── ... (métriques (csv) et graphiques pour l'intersection (png))
│   ├── results_union/
│   │   └── ... (métriques (csv) et graphiques pour l'union (png))
│   └── overlap_analysis/
│       ├── synonym_level/
│       │   └── ... (matrices (csv) et matrices de Jaccard pour le chevauchement par synonyme (png))
│       └── synonym_COMBINATIONS/
│           └── combinations/
│               └── ... (matrices et cartes de chaleur de Jaccard pour le chevauchement par combinaison (png))
└── README.md

📖 Description des Fichiers Clés
Chaque composant du projet a un rôle spécifique :

src/config.py : ⚙️ Le cœur de la configuration ! Ce fichier centralise tous les paramètres globaux du projet. Vous y trouverez les types d'entités et leurs synonymes, les chemins d'accès aux données, le nom du modèle GLiNER, et les seuils par défaut pour la prédiction et la similarité de Jaccard.

src/utils.py : 🛠️ Une boîte à outils essentielle. Ce fichier regroupe des fonctions d'aide utilisées à travers tous les scripts. Il contient des utilitaires pour charger des données JSON, initialiser le modèle GLiNER, et calculer des métriques clés comme l'indice de Jaccard, la précision, le rappel et le score F1.

src/predict_by_synonym.py : 🎯 Prédire par synonyme individuel. Ce script utilise GLiNER pour prédire les entités pour chaque synonyme défini dans config.py. Les résultats bruts (ID de texte, segment, texte de l'entité, étiquette) sont stockés dans outputs/debug_by_synonym.json.

src/predict_combinations.py : 🧩 Prédire avec des combinaisons. Similaire au script précédent, celui-ci prédit les entités en exploitant des combinaisons de synonymes. Il parcourt toutes les associations possibles pour chaque type d'entité, sauvegardant les prédictions dans outputs/debug_combinations.json.

src/evaluate_union.py : 📈 Évaluation par l'union globale. Ce script évalue la performance des prédictions basées sur l'union des segments prédits par différentes combinaisons de synonymes. Il calcule précision, rappel et F1-score (correspondances exactes et partielles avec seuil de Jaccard), et génère les métriques (metrics_set_union.csv) et des graphiques sous outputs/results_union/.

src/evaluate_union_indiv.py : 📊 Contribution des synonymes individuels à l'union. Ce script évalue l'union des prédictions en se concentrant sur la contribution spécifique de chaque synonyme individuel. Les métriques et graphiques sont également sauvegardés dans le répertoire results_union.

src/evaluate_intersection.py : 📉 Évaluation par l'intersection des prédictions. Ce script mesure la performance des prédictions basées sur l'intersection des segments prédits. Il calcule les mêmes métriques (précision, rappel, F1-score avec seuil de Jaccard), enregistrant les résultats dans outputs/results_intersection/metrics_set_intersections.csv et créant des graphiques.

src/overlap_by_synonym.py : 🤝 Analyse du chevauchement par synonyme. Ce script calcule l'indice de Jaccard pour quantifier le chevauchement entre les prédictions générées par des synonymes individuels pour chaque type d'entité. Il produit une matrice de similarité et une carte de chaleur, disponibles dans outputs/overlap_analysis/synonym_level/.

src/overlap_combinations.py : 🔗 Analyse du chevauchement par combinaisons. Ce script analyse le chevauchement entre les prédictions dérivées de différentes combinaisons de synonymes. Il génère des matrices de Jaccard et des cartes de chaleur que vous trouverez dans outputs/overlap_analysis/synonym_COMBINATIONS/combinations/.

data/fulldata.json : 📥 Le jeu de données. Ce répertoire contient le jeu de données d'entrée utilisé pour toutes les opérations de prédiction et d'évaluation des entités.

dataf_factory/data.py : 🏭 Fabrique de données. Ce fichier est probablement utilisé pour la génération ou la manipulation de données d'entrée avant leur utilisation par les scripts principaux.

outputs/ : 📦 Le dossier de tous les résultats. Ce répertoire contient tous les fichiers générés par l'exécution du projet :

debug_by_synonym.json : Prédictions brutes pour chaque synonyme individuel.

debug_combinations.json : Prédictions brutes pour chaque combinaison de synonymes.

results_intersection/ : Fichiers CSV des métriques et graphiques pour les évaluations basées sur l'intersection.

results_union/ : Fichiers CSV des métriques et graphiques pour les évaluations basées sur l'union.

overlap_analysis/ : Sous-répertoires contenant les matrices de Jaccard et les cartes de chaleur pour l'analyse de chevauchement.

🚀 Comment Démarrer
Pour utiliser ce projet, suivez ces étapes :

Cloner le dépôt :

git clone https://github.com/votre-utilisateur/votre-projet.git
cd votre-projet

(Remplacez votre-utilisateur/votre-projet par le chemin réel de votre dépôt.)

Installer les dépendances : (Assurez-vous d'avoir Python installé)

pip install -r requirements.txt # Si vous avez un fichier requirements.txt
# Ou installez manuellement les bibliothèques comme transformers, scikit-learn, matplotlib, pandas, etc.

Exécuter les scripts :
Vous pouvez exécuter les scripts dans l'ordre pour générer les prédictions, les évaluations et les analyses de chevauchement. Par exemple :

python src/predict_by_synonym.py
python src/predict_combinations.py
python src/evaluate_union.py
# ... et ainsi de suite

N'oubliez pas de configurer vos chemins et paramètres dans src/config.py avant de lancer les scripts.

🤝 Contribution
Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce projet, n'hésitez pas à ouvrir une issue ou à soumettre une pull request.

📄 Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails. (N'oubliez pas de créer ce fichier si vous n'en avez pas encore un !)

📧 Contact
Pour toute question ou suggestion, n'hésitez pas à me contacter : votre.email@example.com (N'oubliez pas de remplacer ceci par votre adresse email !)
