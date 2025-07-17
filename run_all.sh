#!/bin/bash

echo "🚀 Démarrage du pipeline GLiNER Synonym Evaluation..."

# Inclure le répertoire courant dans PYTHONPATH pour que 'src' soit reconnu
export PYTHONPATH=$(pwd)

# Étapes du pipeline
scripts=(
    "src/predict_by_synonym.py"
    "src/predict_combinations.py"
    "src/evaluate_union.py"
    "src/evaluate_intersection.py"
    "src/overlap_by_synonym.py"
    "src/overlap_combinations.py"
)

# Seuil par défaut pour l'indice de Jaccard
JACCARD_THRESHOLD=0.5

for script in "${scripts[@]}"; do
    echo ""
    echo "▶️  Exécution de : $script"
    
    if [[ "$script" == *"evaluate_union.py" ]] || [[ "$script" == *"evaluate_intersection.py" ]]; then
        python "$script" --threshold $JACCARD_THRESHOLD
    else
        python "$script"
    fi

    if [[ $? -ne 0 ]]; then
        echo "❌ Erreur lors de l'exécution de $script. Arrêt du pipeline."
        exit 1
    else
        echo "✅ $script terminé avec succès."
    fi
done

echo ""
echo "🎉 Pipeline complet terminé avec succès."
